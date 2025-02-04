import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import get_scheduler, AdamW
from dataset import MathDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

checkpoint = "HuggingFaceTB/SmolLM-135M-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)


train_dataset = MathDataset(tokenizer, "./GMS8K/train.jsonl", "./MATH/train")
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
optim = AdamW(model.parameters(), lr=1e-5)

num_epochs = 1
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optim,
    num_warmup_steps=16,
    num_training_steps=num_training_steps,
)

simulated_batch_size = 8

pbar = tqdm(range(num_training_steps))
for epoch in range(num_epochs):
    bi = 0
    for batch in train_loader:

        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs[0]

        loss.backward()

        pbar.update(1)
        pbar.set_description(f"train_loss: {loss.item():.5f}")
        
        if ((bi + 1) % simulated_batch_size == 0) or (bi + 1 == len(train_loader)):
            optim.step()
            lr_scheduler.step()
            optim.zero_grad()
            model.save_pretrained("simple_ft_checkpoint/")

        if bi % 100 == 0:
            model.save_pretrained(f"simple_ft_checkpoint{int(bi / 100)}/")

        bi += 1