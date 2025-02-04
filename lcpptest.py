from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_scheduler

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)


llm = Llama.from_pretrained(
    repo_id="https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct-GGUF/blob/main/smollm2-360m-instruct-q8_0.gguf",
    n_gpu_layers=-1,
)


prompt = "Hello"

messages = [{"role": "user", "content": prompt}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False
)

output = llm(
      prompt, # Prompt
      max_tokens=1024, # Generate up to 32 tokens, set to None to generate up to the end of the context window
      stop=["<|im_end|>"], # Stop generating just before the model would generate a new question
      echo=True # Echo the prompt back in the output
) # Generate a completion, can also call create_completion

print(output)