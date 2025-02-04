# Tiny R1

Welcome to Tiny R1! The goal of this repository is to replicate advanced thinking techniques that DeepSeek R1 and OpenAI o1/o3 exhibit using smaller models and pure RL (no distillation!).
The hope is that good implementations of RL (like GRPO/PPO), reward models, and finetuning can be available to anyone.

## How to Use
Right now, the state of Tiny R1 is training a math-logic model using the MATH and GSM8K datasets. If you want to run the training scripts, you must download both datasets and place in the same directory as the traning script. 
For GSM8K, make sure the train jsonl is in a folder called GSM8K. For MATH, ensure that the train directory with all of the JSONs is in a folder called MATH.

[MATH](https://www.kaggle.com/datasets/mathurinache/math-dataset)
[GSM8K](https://github.com/openai/grade-school-math) (grade_school_math/data)

# What does it do, and does it work?

Right now, I have implemented three different kinds of finetuning (normal ft, PPO, and GRPO). I am fine tuning HuggingFace's smol-360m LLM, a very tiny model. 
Due to resource limitations it gets hard to train beyond that (although future trials will be using 1b models). Unfortunately, out of the trials I did, none seem to have improved significantly. I trained each method using LoRA of rank 32 on a single H100 for ~2 hours. 

This is actually an expected result, as DeepSeek, in their R1 paper, has already shown that a rule-based reward model along with their GRPO training schema doesn't make any noticable difference. However there are a few next steps that I want to take to see if I am able to improve such a small model.

# Next Steps
### Add per-token rewards
For efficiency, I used an outcome-based reward model. This reward was applied to tokens for a single output. This is likely making it much harder for reinforcement techniques to learn. Instead, I want to transition to a process-based reward model which can provide a reward for each token. This will probably incorporate the reasoning provided by the GSM8K and MATH dataset.

### Model-based Reward Model
Currently, I am using a purely rule based model. While DeepSeek showed that this can work with their DeepSeek-r1-zero, they also showed that when applied to a pretrained model it showed no significant improvement. While the usage of a rule-based model might not be the culprint, I think that it will be worth it to implement a similar reward model to the one that DeepSeek discusses in their DeepSeekMath paper. Of course, this comes with significant resource overhead.

### More Tokens, More Compute, More Parameters
As always, throwing more data, more parameters and more processing power can improve a model.

### Higher Rank LoRA / no LoRA
In order to speed up training, I decided to add LoRA adapters. However, it isn't clear whether this is impeding on RL's ability, or if the rank is too low for the policy to learn what it needs to.
