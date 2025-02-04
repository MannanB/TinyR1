import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, AdamW
from dataset import MathDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from peft import LoraConfig, get_peft_model, TaskType

checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
model.train()
print("hi!")
print(model)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,   # Specify the task type
    r=8,                           # Rank of the decomposition matrix
    lora_alpha=32,                 # Scaling factor
    lora_dropout=0.1,              # Dropout probability for LoRA layers
    target_modules=["q_proj", "v_proj", "up_proj", "down_proj"]  # List of module names to inject LoRA adapters
)

# Wrap the original model with the LoRA adapter
model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

# --- SETUP DATA, OPTIMIZER, AND SCHEDULER ---
train_dataset = MathDataset(tokenizer, "./GSM8K/train.jsonl", "./MATH/train")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
optim = AdamW(model.parameters(), lr=1e-5)

num_epochs = 1
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optim,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

simulated_batch_size = 8

pbar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    bi = 0
    for batch in train_loader:
        # Move batch to the GPU
        batch = {k: v.to(device) for k, v in batch.items()}
        # Forward pass with labels (for language modeling)
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss  

        loss.backward()

        pbar.update(1)
        pbar.set_description(f"train_loss: {loss.item():.5f}")

        # Gradient accumulation step
        if ((bi + 1) % simulated_batch_size == 0) or (bi + 1 == len(train_loader)):
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()
            # Save the LoRA-adapted model checkpoint
            model.save_pretrained("simple_ft_checkpoint/")

        torch.cuda.empty_cache()
        del batch, outputs, loss

        bi += 1
