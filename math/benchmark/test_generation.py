"""
Quick generation test to verify gpt-oss-120b loads and generates correctly.
Run with: /data/work/irdali.durrani/miniconda3/bin/python math/benchmark/test_generation.py
"""

import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai/gpt-oss-120b"
CACHE_DIR = "/data/cache/huggingface/hub"

PROBLEMS = [
    "What is 2 + 2?",
    "Find all real solutions to x^2 - 5x + 6 = 0.",
    "What is the derivative of x^3 + 2x?",
]

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)

print(f"Loading model (this may take a minute)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print(f"Model loaded on: {next(model.parameters()).device}\n")

for problem in PROBLEMS:
    print(f"{'='*60}")
    print(f"PROBLEM: {problem}")

    messages = [{"role": "user", "content": problem}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    # The response contains analysis (thinking) and final sections separated by "assistantfinal"
    # We only want the final answer part
    if "assistantfinal" in generated:
        final = generated.split("assistantfinal", 1)[1].strip()
    else:
        final = generated.strip()
    print(f"THINKING: {generated.split('assistantfinal')[0].replace('analysis', '').strip()}")
    print(f"FINAL:\n{final}\n")
