
import torch
import json
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = '/home/tejag/projects/dart/ai/tensor/models/llm/gpt2'
output_file = 'gpt2_test_data.json'

print(f"Loading model from {model_path}...")
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

prompts = [
    "Hello, my name is",
    "The quick brown fox",
    "Artificial intelligence is",
    "Once upon a time in a",
    "To be or not to be"
]

test_data = []

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=20, 
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    test_data.append({
        "prompt": prompt,
        "response": response_text
    })

script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, output_file)

with open(output_path, 'w') as f:
    json.dump(test_data, f, indent=2)

print("Done!")
