import torch, re

def get_all_numbers_in_text(text):
    return list(set(re.findall(r"[-+]?\d*\.\d+|\d+", text)))

def reward_model(token_list, correct_answer, correct_reasoning):
    reward = 0.0
    decoded_tokens = []
    answer_tokens = []
    thinking_tokens = []
    counts = {"<think>": 0, "</think>": 0, "<answer>": 0, "</answer>": 0, "<|im_end|>": 0}
    in_thinking = False
    in_answering = False
    for tkn in token_list:
        tkn = tkn.strip().lower()
        if not tkn: continue
        decoded_tokens.append(tkn)
        if tkn == ">":
            if len(decoded_tokens) >= 4 and decoded_tokens[-3] == "<" or decoded_tokens[-3] == "</":
                tag = decoded_tokens[-2]
                bracket = decoded_tokens[-3]
                decoded_tokens[-1] = f"{bracket}{tag}>"
    for tkn in decoded_tokens:
        if in_thinking: thinking_tokens.append(tkn)
        if in_answering: answer_tokens.append(tkn)
        if tkn == "<think>": in_thinking = True
        if tkn == "</think>": in_thinking = False
        if tkn == "<answer>": in_answering = True
        if tkn == "</answer>": in_answering = False
        counts[tkn] = counts.get(tkn, 0) + 1
    if counts.get("<think>", 0) == 1: reward += 0.5
    if counts.get("</think>", 0) == 1: reward += 0.5
    if counts.get("<answer>", 0) == 1: reward += 0.5
    if counts.get("</answer>", 0) == 1: reward += 0.5
    if "<|im_end|>" in decoded_tokens: reward += 0.5
    if "<|im_end|>" in decoded_tokens and "</answer>" in decoded_tokens and decoded_tokens.index("<|im_end|>") == decoded_tokens.index("</answer>") + 1:
        reward += 0.5
    if "<think>" in decoded_tokens and "istant" in decoded_tokens and decoded_tokens.index("<think>") == decoded_tokens.index("istant") + 1:
        reward += 0.5
    if "<answer>" in decoded_tokens and "</think>" in decoded_tokens and decoded_tokens.index("<answer>") == decoded_tokens.index("</think>") + 1:
        reward += 0.5
    if counts.get("</answer>", 0) == 1 and len(answer_tokens) > 0:
        reward += 1.0
    if str(correct_answer) in "".join(answer_tokens).lower():
        reward += 10.0
    if counts.get("</think>", 0) == 1 and len(thinking_tokens) > 10:
        reward += 1.0
    correct_nums = get_all_numbers_in_text(correct_reasoning)
    gen_nums = get_all_numbers_in_text("".join(decoded_tokens))
    for num in correct_nums:
        if num in gen_nums:
            reward += 0.3
    return reward

def sample_logit(single_logit, temperature=0.2, top_p=0.9):
    scaled_logits = single_logit / temperature

    probs = torch.softmax(scaled_logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)

    filtered_probs = sorted_probs / sorted_probs.sum()

    new_probs = torch.zeros_like(probs)
    new_probs.scatter_(dim=-1, index=sorted_indices, src=filtered_probs)

    sampled_token_id = torch.multinomial(new_probs, num_samples=1)

    one_hot = torch.zeros_like(new_probs)
    one_hot.scatter_(dim=-1, index=sampled_token_id, value=1.0)
    return one_hot

def decode_single_logit(single_logit, tokenizer):
    token_id = sample_logit(single_logit).argmax().unsqueeze(0)
    return tokenizer.decode(token_id)

def sample_token_from_logits(logits, temperature=0.2, top_p=0.9):
    one_hot = sample_logit(logits[0], temperature, top_p)
    token_id = one_hot.argmax().item()
    return token_id

def decode_entire_logits(scores, tokenizer, temperature=0.2, top_p=0.9):
    token_ids = []
    for score in scores:
        # Sample a token from the score vector.
        token_id = sample_token_from_logits(score, temperature, top_p)
        print("appending token id", token_id)
        token_ids.append(token_id)
    # Decode the entire sequence of token IDs.
    decoded_text = [tokenizer.decode(t) for t in token_ids]
    return decoded_text