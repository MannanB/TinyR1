import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler
from torch.utils.data import DataLoader
from dataset import MathDataset
import re
import numpy as np
import tqdm
import copy

from utils import reward_model

from peft import LoraConfig, get_peft_model, TaskType


# Hyperparameters
LR = 1e-5
MAX_NEW_TOKENS = 1024
GAMMA = 0.99
LAM = 0.95
EPS_CLIP = 0.1
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
NUM_EPOCHS = 1
SIM_BATCH_SIZE = 4

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Specify the task type
    r=32,                           # Rank of the decomposition matrix
    lora_alpha=64,                 # Scaling factor
    lora_dropout=0.05,              # Dropout probability for LoRA layers
    target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]  # List of module names to inject LoRA adapters (adjust as needed)
)

# Wrap the original model with the LoRA adapter
model = get_peft_model(model, lora_config)
# Optional: print trainable parameters to verify only LoRA parameters are being optimized
model.print_trainable_parameters()


model.config.output_hidden_states = True  # so we can compute value estimates



# A simple value head that maps last hidden states to a scalar value per token.
class ValueHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, hidden_states):
        return self.linear(hidden_states).squeeze(-1)

value_head = ValueHead(model.config.hidden_size).to(device)

train_dataset = MathDataset(tokenizer, "./GMS8K/train.jsonl", "./MATH/train")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
optim = AdamW(list(model.parameters()) + list(value_head.parameters()), lr=LR)
num_training_steps = NUM_EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optim, num_warmup_steps=16, num_training_steps=num_training_steps)



def compute_gae(rewards, values, gamma, lam):
    # rewards and values: tensors of shape (T,) for one sequence; assume terminal state's value = 0.
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

# Create an "old" model copy for PPO KL calculations.
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

        # Generate continuations.
        with torch.no_grad():
            gen_sequences = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
            )
        # Use only the generated part.
        gen_part = gen_sequences[:, prompt_len:]
        # Decode tokens to compute reward.
        gen_tokens = [tokenizer.decode(x) for x in gen_part[0].tolist()]

        reward_val = reward_model(gen_tokens, batch["correct_answer"][0], batch["reasoning"][0])
        reward_tensor = torch.zeros((gen_part.size(0), gen_part.size(1)), device=device)
        reward_tensor = reward_tensor + (reward_val / gen_part.size(1))  # reward only at final token

        # Teacher-forcing pass on generated sequence.
        attn = torch.ones_like(gen_sequences).to(device)
        outputs_new = model(gen_sequences, attention_mask=attn, output_hidden_states=True, return_dict=True)
        logits_new = outputs_new.logits  # (B, L, V)
        hs_new = outputs_new.hidden_states[-1]  # (B, L, H)
        new_log_probs = F.log_softmax(logits_new, dim=-1)
        new_log_probs = new_log_probs[:, prompt_len:, :]  # (B, T, V)
        # Gather log probs for actions.
        gen_actions = gen_sequences[:, prompt_len:]
        new_action_log_probs = new_log_probs.gather(2, gen_actions.unsqueeze(-1)).squeeze(-1)  # (B, T)
        # Entropy.
        entropy = -(new_log_probs * new_log_probs.exp()).sum(-1).mean()


        with torch.no_grad():
            outputs_old = old_model(gen_sequences, attention_mask=attn, output_hidden_states=True, return_dict=True)
            logits_old = outputs_old.logits
        old_log_probs = F.log_softmax(logits_old, dim=-1)[:, prompt_len:, :]
        old_action_log_probs = old_log_probs.gather(2, gen_actions.unsqueeze(-1)).squeeze(-1)


        # Value predictions.
        values = value_head(hs_new)[:, prompt_len:]  # (B, T)

        # Compute advantages & returns per sequence (looping over batch if needed).
        advantages_list = []
        returns_list = []
        for i in range(values.size(0)):
            adv, ret = compute_gae(reward_tensor[i], values[i], GAMMA, LAM)
            advantages_list.append(adv)
            returns_list.append(ret)
        advantages = torch.stack(advantages_list, dim=0)
        returns = torch.stack(returns_list, dim=0)

        # PPO loss.
        log_ratio = new_action_log_probs - old_action_log_probs
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

        loss.backward()
        if (bi + 1) % SIM_BATCH_SIZE == 0 or (bi + 1) == len(train_loader):
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()
            model.save_pretrained("ppo2_checkpoint/")
            torch.cuda.empty_cache()
            with open("./debug.txt", "w", errors="ignore") as f:
                f.write(tokenizer.decode(gen_sequences[0], skip_special_tokens=False) + "\n" + f"loss: {loss.item():.5f}, reward: {reward_val:.5f}, adv: {advantages.mean().item():.5f}, val: {value_loss.item():.5f}, ent: {entropy.item():.5f}")


        if (bi + 1) % SIM_BATCH_SIZE * 3 == 0 or (bi + 1) == len(train_loader):
            old_model.load_state_dict(model.state_dict())


        if bi % 100 == 0:
            model.save_pretrained(f"ppo2_checkpoint_{int(bi/100)}/")

        global_step += 1
        pbar.update(1)
        pbar.set_description(f"loss: {loss.item():.5f}, reward: {reward_val:.5f}, adv: {advantages.mean().item():.5f}, val: {value_loss.item():.5f}, ent: {entropy.item():.5f}")



        del input_ids, attention_mask, gen_sequences, outputs_new, outputs_old, logits_new, logits_old
        torch.cuda.empty_cache()

pbar.close()