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

# NOTE: This PPO implementation is INCOMPLETE as KL-divergence is not implemented.

LR = 1e-5
MAX_NEW_TOKENS = 1024
GAMMA = 0.99
LAM = 0.95
EPS_CLIP = 0.1
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
NUM_EPOCHS = 1
BATCH_SIZE = 12  # overall batch size
ACCUM_MINI_BATCH_SIZE = 4  # mini batch size for teacher-forcing gradient accumulation

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto").to(device)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Specify the task type
    r=64,                           # Rank of the decomposition matrix
    lora_alpha=128,                  # Scaling factor
    lora_dropout=0.05,              # Dropout probability for LoRA layers
    target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]
)
# Wrap the original model with the LoRA adapter.
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable output of hidden states so we can compute value estimates.
model.config.output_hidden_states = True


class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states):
        return self.linear(hidden_states).squeeze(-1)

value_head = ValueHead(model.config.hidden_size).to(device).to(torch.bfloat16) 

train_dataset = MathDataset(tokenizer, "./GSM8K/train.jsonl", "./MATH/train")
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Set up optimizer and scheduler.
optim = AdamW(list(model.parameters()) + list(value_head.parameters()), lr=LR)
num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optim,
                             num_warmup_steps=16, num_training_steps=num_training_steps)


def compute_gae(rewards, values, gamma, lam):
    # rewards and values are 1D tensors of length T (tokens for one sequence).
    T = rewards.size(0)
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(T)):
        next_val = 0 if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns

old_model = copy.deepcopy(model)
old_model.eval()
old_model.to(device)

model.train()
value_head.train()

pbar = tqdm.tqdm(total=num_training_steps)
global_step = 0

for epoch in range(NUM_EPOCHS):
    for bi, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        prompt_len = input_ids.size(1)
        
        with torch.no_grad():
            gen_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                eos_token_id=tokenizer("<|im_end|>")["input_ids"][0]
            )
        # gen_sequences shape: (B, L_total)
        gen_part = gen_sequences[:, prompt_len:]  # generated tokens only

        reward_tensor = torch.zeros(gen_part.size(), device=device)
        for i in range(input_ids.size(0)):
            # Decode generated tokens for sample i.
            gen_tokens = [tokenizer.decode([token_id]) for token_id in gen_part[i].tolist()]
            reward_val = reward_model(gen_tokens, batch["correct_answer"][i], batch["reasoning"][i])
            # Spread the reward evenly over all generated tokens.
            reward_tensor[i] = reward_val / gen_part.size(1)
        
        loss_total = 0.0
        B = gen_sequences.size(0) 
        for start in range(0, B, ACCUM_MINI_BATCH_SIZE):
            end = start + ACCUM_MINI_BATCH_SIZE
            mini_gen_sequences = gen_sequences[start:end]  # (mini_B, L_total)
            mini_attn = torch.ones_like(mini_gen_sequences).to(device)
            
            mini_outputs_new = model(
                mini_gen_sequences,
                attention_mask=mini_attn,
                output_hidden_states=True,
                return_dict=True,
            )
            mini_logits_new = mini_outputs_new.logits  # (mini_B, L_total, V)
            mini_hs_new = mini_outputs_new.hidden_states[-1]  # (mini_B, L_total, H)
            mini_new_log_probs = F.log_softmax(mini_logits_new, dim=-1)[:, prompt_len:, :]  # (mini_B, T, V)
            mini_gen_actions = mini_gen_sequences[:, prompt_len:]  # (mini_B, T)
            mini_new_action_log_probs = mini_new_log_probs.gather(2, mini_gen_actions.unsqueeze(-1)).squeeze(-1)
            mini_entropy = -(mini_new_log_probs * mini_new_log_probs.exp()).sum(-1).mean()
            
            with torch.no_grad():
                mini_outputs_old = old_model(
                    mini_gen_sequences,
                    attention_mask=mini_attn,
                    output_hidden_states=True,
                    return_dict=True,
                )
            mini_logits_old = mini_outputs_old.logits
            mini_old_log_probs = F.log_softmax(mini_logits_old, dim=-1)[:, prompt_len:, :]
            mini_old_action_log_probs = mini_old_log_probs.gather(2, mini_gen_actions.unsqueeze(-1)).squeeze(-1)
        
            mini_values = value_head(mini_hs_new)[:, prompt_len:]  # (mini_B, T)
            
            # Compute GAE per mini sample
            mini_advantages_list = []
            mini_returns_list = []
            for i in range(mini_values.size(0)):
                adv, ret = compute_gae(reward_tensor[start + i], mini_values[i], GAMMA, LAM)
                mini_advantages_list.append(adv)
                mini_returns_list.append(ret)
            mini_advantages = torch.stack(mini_advantages_list, dim=0)  # (mini_B, T)
            mini_returns = torch.stack(mini_returns_list, dim=0)          # (mini_B, T)
            
            mini_log_ratio = mini_new_action_log_probs - mini_old_action_log_probs
            mini_ratio = torch.exp(mini_log_ratio)
            mini_surr1 = mini_ratio * mini_advantages
            mini_surr2 = torch.clamp(mini_ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * mini_advantages
            mini_policy_loss = -torch.min(mini_surr1, mini_surr2).mean()
            mini_value_loss = F.mse_loss(mini_values, mini_returns)
            mini_loss = mini_policy_loss + VALUE_COEF * mini_value_loss - ENTROPY_COEF * mini_entropy
            
            mini_loss.backward()
            loss_total += mini_loss.item()
            torch.cuda.empty_cache()

        optim.step()
        lr_scheduler.step()
        optim.zero_grad()

        old_model.load_state_dict(model.state_dict())

        model.save_pretrained("ppo2_checkpoint/")
        with open("./debug.txt", "w", errors="ignore") as f:
            debug_text = (
                tokenizer.decode(gen_sequences[0], skip_special_tokens=False) + "\n" +
                f"loss: {loss_total:.5f}, reward: {reward_tensor[0].mean().item():.5f}, " +
                f"adv: {mini_advantages.mean().item():.5f}, value_loss: {mini_value_loss.item():.5f}, " +
                f"entropy: {mini_entropy.item():.5f}"
            )
            f.write(debug_text)

        global_step += 1
        pbar.update(1)
        pbar.set_description(
            f"loss: {loss_total:.5f}, reward: {reward_tensor[0].mean().item():.5f}, " +
            f"adv: {mini_advantages.mean().item():.5f}, value_loss: {mini_value_loss.item():.5f}, " +
            f"entropy: {mini_entropy.item():.5f}"
        )

        del input_ids, attention_mask, gen_sequences, mini_outputs_new, mini_outputs_old
        torch.cuda.empty_cache()

pbar.close()
