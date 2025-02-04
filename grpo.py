import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
from torch.utils.data import DataLoader
from dataset import MathDataset
import numpy as np
import tqdm, time
import copy

from utils import reward_model

from peft import LoraConfig, get_peft_model, TaskType


LR = 1e-5
MAX_NEW_TOKENS = 512
EPS_CLIP = 0.1
ENTROPY_COEF = 0.01
NUM_EPOCHS = 1
NUM_GRPO_OUTPUTS = 16 
ACCUM_MINI_BATCH_SIZE = 8

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# Load the model and move it to the primary device.
base_model = AutoModelForCausalLM.from_pretrained(
    checkpoint, torch_dtype=torch.bfloat16, device_map="auto"
).to(device)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]
)
# Apply LoRA *before* wrapping in DataParallel.
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

# Now wrap the model with DataParallel if more than one GPU is available.
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
    model = nn.DataParallel(model)


train_dataset = MathDataset(tokenizer, "./GMS8K/train.jsonl", "./MATH/train")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=LR)
num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optim,
                             num_warmup_steps=16, num_training_steps=num_training_steps)

if isinstance(model, nn.DataParallel):
    old_base_model = copy.deepcopy(model.module)
else:
    old_base_model = copy.deepcopy(model)
old_model = old_base_model.to(device)
old_model.eval()
if torch.cuda.device_count() > 1:
    old_model = nn.DataParallel(old_model)

model.train()

pbar = tqdm.tqdm(total=num_training_steps)
global_step = 0

for epoch in range(NUM_EPOCHS):
    for bi, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        batch_size = input_ids.size(0)
        prompt_len = input_ids.size(1)

        t1 = time.time()
        with torch.no_grad():
            # If the model is wrapped in DataParallel, use model.module for generation.
            if isinstance(model, nn.DataParallel):
                gen_sequences = model.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    num_return_sequences=NUM_GRPO_OUTPUTS,
                    eos_token_id=tokenizer("<|endoftext|>")["input_ids"][0]
                )
            else:
                gen_sequences = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    num_return_sequences=NUM_GRPO_OUTPUTS,
                    eos_token_id=tokenizer("<|endoftext|>")["input_ids"][0]
                )
        t2 = time.time()
        total_generated = gen_sequences.size(0)  # batch_size * NUM_GRPO_OUTPUTS
        pbar.set_description(f"Generation Done in {t2-t1:.2f}s with {NUM_GRPO_OUTPUTS} completions")


        rewards = torch.zeros((batch_size, NUM_GRPO_OUTPUTS), device=device, dtype=torch.float)
        for i in range(batch_size):
            for j in range(NUM_GRPO_OUTPUTS):
                idx = i * NUM_GRPO_OUTPUTS + j
                gen_part = gen_sequences[idx, prompt_len:]
                gen_tokens = [tokenizer.decode([token_id]) for token_id in gen_part.tolist()]
                reward_val = reward_model(gen_tokens, batch["correct_answer"][i], batch["reasoning"][i])
                rewards[i, j] = reward_val

        # NOTE: rewards are the SAME for all tokens in a sequence.
        # NOTE: KL-divergence has not been implemented
        # Compute advantages using rewards as defined in DeepSeekMath

        advantages = torch.zeros_like(rewards)
        for i in range(batch_size):
            sample_rewards = rewards[i]
            mean = sample_rewards.mean()
            std = sample_rewards.std() + 1e-6
            advantages[i] = (sample_rewards - mean) / std

        flat_rewards = rewards.view(-1)
        flat_advantages = advantages.view(-1)

        # using teacher forcing + mini batches (to get around vram limits)
        loss_total = 0.0
        for start in tqdm.tqdm(range(0, total_generated, ACCUM_MINI_BATCH_SIZE), desc="Backpropagating", leave=False):
            end = start + ACCUM_MINI_BATCH_SIZE
            mini_gen_sequences = gen_sequences[start:end]
            mini_attn = torch.ones_like(mini_gen_sequences).to(device)
            
            outputs_new = model(mini_gen_sequences,
                                attention_mask=mini_attn,
                                output_hidden_states=True,
                                return_dict=True)
            logits_new = outputs_new.logits
            mini_new_log_probs = F.log_softmax(logits_new, dim=-1)[:, prompt_len:, :]
            mini_gen_actions = mini_gen_sequences[:, prompt_len:]
            mini_new_action_log_probs = mini_new_log_probs.gather(2, mini_gen_actions.unsqueeze(-1)).squeeze(-1)
            mini_entropy = -(mini_new_log_probs * mini_new_log_probs.exp()).sum(-1).mean()

            with torch.no_grad():
                mini_outputs_old = old_model(mini_gen_sequences,
                                             attention_mask=mini_attn,
                                             output_hidden_states=True,
                                             return_dict=True)
            mini_logits_old = mini_outputs_old.logits
            mini_old_log_probs = F.log_softmax(mini_logits_old, dim=-1)[:, prompt_len:, :]
            mini_old_action_log_probs = mini_old_log_probs.gather(2, mini_gen_actions.unsqueeze(-1)).squeeze(-1)

            mini_advantages = flat_advantages[start:end]
            mini_advantages_expanded = mini_advantages.unsqueeze(1)
            mini_log_ratio = mini_new_action_log_probs - mini_old_action_log_probs
            mini_ratio = torch.exp(mini_log_ratio)
            mini_surr1 = mini_ratio * mini_advantages_expanded
            mini_surr2 = torch.clamp(mini_ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * mini_advantages_expanded
            mini_policy_loss = -torch.min(mini_surr1, mini_surr2).mean()
            
            mini_loss = mini_policy_loss - ENTROPY_COEF * mini_entropy

            mini_loss.backward()
            loss_total += mini_loss.item()

            torch.cuda.empty_cache()
        
        optim.step()
        lr_scheduler.step()
        optim.zero_grad()

        model.save_pretrained("grpo3_checkpoint/")

        debug_text = (
            tokenizer.decode(gen_sequences[0], skip_special_tokens=False) + "\n" +
            f"loss: {loss_total:.5f}, reward: {flat_rewards[0].item():.5f}, " +
            f"advantage: {flat_advantages[0].item():.5f}"
        )
        with open("./debug.txt", "w", errors="ignore") as f:
            f.write(debug_text)

        if (bi % 2) == 0:
            # Update the old model from the underlying module.
            current_base = model.module if isinstance(model, nn.DataParallel) else model
            old_model.load_state_dict(current_base.state_dict())

        if bi % 100 == 0:
            model.save_pretrained(f"grpo3_checkpoint_{int(bi/100)}/")

        global_step += 1
        pbar.update(1)
        pbar.set_description(
            f"loss: {loss_total:.5f}, avg_reward: {flat_rewards.mean().item():.5f}, " +
            f"max_adv: {flat_advantages.max().item():.5f}, max_reward: {flat_rewards.max().item():.5f}"
        )

        torch.cuda.empty_cache()

pbar.close()
