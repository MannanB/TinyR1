import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import decode_single_logit, decode_entire_logits
from dataset import GMS8K, MATH

# gms8k = GMS8K("./GMS8K/test.jsonl")
# data = gms8k.process()
math = MATH("./MATH/test")
data = math.process()
n = 10

parent_checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda"
A = "HuggingFaceTB/SmolLM2-360M-Instruct"
B = "./grpo_checkpoint"

tokenizer = AutoTokenizer.from_pretrained(parent_checkpoint)


for checkpoint in ["./grpo_checkpoint", "HuggingFaceTB/SmolLM2-360M-Instruct", "./ppo_checkpoint"]:
    num_correct = 0

    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    for i in range(20, 20+n):
        question = data[i]["question"]

        THINK_PROMPT = f"""You will be asked a math question. You first think about the reasing process and the steps to solve the problem and then provide the user with an answer.
        The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>
        Question: {question}
        """

        messages = [{"role": "user", "content": THINK_PROMPT}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False) + ""

        inputs = tokenizer(input_text, return_tensors="pt")
        # print([tokenizer.decode(t) for t in inputs["input_ids"]])

        
        output_tokens = model.generate(inputs["input_ids"].to(device), max_new_tokens=512, temperature=0.2, top_p=0.9, do_sample=True, repetition_penalty=1.2, attention_mask=inputs["attention_mask"].to(device), num_return_sequences=8, eos_token_id=tokenizer("<|im_end|>")["input_ids"][0])
        tokenizer.decode(output_tokens[0])
        for ot in output_tokens:
            if data[i]["answer"] in tokenizer.decode(ot):
                num_correct += 1
                break
    print(f"Checkpoint: {checkpoint}. Correct: {num_correct}/{n}")
