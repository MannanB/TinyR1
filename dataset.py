import json
import re
import os

import torch

class GSM8K:
    def __init__(self, path):
        self.path = path

        with open(self.path) as f:
            self.jsonl = [json.loads(line) for line in f.readlines() if line]

    def process(self):
        remove_subcomp = re.compile(r"<<[^<>]*>>")
        extract_answer = re.compile(r"#### (\-?[0-9\.\,]+)")

        processed = []

        for item in self.jsonl:
            pitem = {"question": "", "reasoning": "", "answer": "", "model_resp": ""}
            pitem["question"] = item["question"]
            pitem["reasoning"] = remove_subcomp.sub("", item["answer"])
            answer = extract_answer.search(item["answer"])
            if not answer:
                continue
            pitem["answer"] = answer.group(1).strip().replace(",", "")
            model_resp = "<think>\n" + pitem["reasoning"] + "\n</think>\n<answer>\n" + pitem["answer"] + "\n</answer>"
            pitem["model_resp"] = model_resp
            processed.append(pitem)
        
        return processed
        
    def __len__(self):
        return len(self.jsonl)
    
class MATH:
    def __init__(self, path):
        self.path = path
        self.jsonl = []

        for root, dirs, files in os.walk(self.path):
            for file in files:
                with open(os.path.join(root, file)) as f:
                    self.jsonl.append(json.load(f))

    def process(self):
        extract_answer = re.compile(r'\\boxed{(.*?)}')

        processed = []

        for item in self.jsonl:
            pitem = {"question": "", "reasoning": "", "answer": "", "model_resp": ""}
            pitem["question"] = item["problem"]
            answer = extract_answer.search(item["solution"])
            if not answer:
                continue
            pitem["answer"] = answer.group(1).strip().replace(",", "")

            pitem["reasoning"] = extract_answer.sub(r"\1", item["solution"]) #+ "\n#### " + pitem["answer"]

            model_resp = "<think>\n" + pitem["reasoning"] + "\n</think>\n<answer>\n" + pitem["answer"] + "\n</answer>"
            pitem["model_resp"] = model_resp
            processed.append(pitem)

        return processed
    
    def __len__(self):
        return len(self.jsonl)
    

THINK_PROMPT = """You will be asked a math question. You first think about the reasing process and the steps to solve the problem and then provide the user with an answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>
Question: {question}
"""

class MathDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, gsm8k_path, math_path):

        self.gms8k = GSM8K(gsm8k_path).process()
        self.math = MATH(math_path).process()

        self.all_data = self.gms8k + self.math

        apply_ct = lambda content, role: tokenizer.apply_chat_template([{"role": role, "content": content}], tokenize=False)

        self.qns_text = [apply_ct(THINK_PROMPT.replace("{question}", ex["question"]), "user") for ex in self.all_data]
        self.ans_text = [apply_ct(ex["model_resp"], "assistant")  for ex in self.all_data]
        self.real_answers = [ex["answer"] for ex in self.all_data]
        self.qns_i = tokenizer(self.qns_text, padding=False)
        self.ans_i = tokenizer(self.ans_text, padding=False)
        self.len_prefix = len(tokenizer("<|im_start|>assistant\n<think>\n", return_tensors="pt")["input_ids"])

        self.qns = []
        self.ans = []
        for i in range(len(self.qns_i["input_ids"])):
            qn = self.qns_i["input_ids"][i]
            ans = self.ans_i["input_ids"][i]
            if len(qn) + len(ans) > 2048:
                continue
            self.qns.append(qn)
            self.ans.append(ans)

        self.max_len = max(
            [
                len(self.qns[i]) + len(self.ans[i])
                for i in range(len(self.qns))
            ]
        )
        self.total_tokens = sum([len(self.qns[i]) + len(self.ans[i]) for i in range(len(self.qns))])
        print(f"Max tokens: {self.max_len}. Number of Samples: {len(self.qns)}. Total tokens: {self.total_tokens}")

    def __len__(self):
        return len(self.qns)

    def __getitem__old(self, idx):
        qn_tokens = self.qns[idx]
        ans_tokens = self.ans[idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([0] * (len(qn_tokens)))
            + ([1] * (len(ans_tokens)))
            + ([0] * len(pad_tokens))
        )
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask, reasoning=self.all_data[idx]["reasoning"], correct_answer=self.real_answers[idx])

    def __getitem__(self, idx):
        qn_tokens = self.qns[idx]
        ans_tokens = self.ans[idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens #+ ans_tokens + pad_tokens
        mask = (
            ([1] * (len(qn_tokens)))
            # + ([1] * (len(ans_tokens)))
            # + ([0] * len(pad_tokens))
        )
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask, reasoning=self.all_data[idx]["reasoning"], correct_answer=self.real_answers[idx])
