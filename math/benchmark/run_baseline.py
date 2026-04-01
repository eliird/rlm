"""
Baseline benchmark: run gpt-oss-120b on the combined test set.

Outputs results to math/benchmark/results/baseline.parquet with columns:
  problem, solution, answer, source, predicted, correct

Run from repo root:
  .venv/bin/python math/benchmark/run_baseline.py [--limit N] [--batch-size N] [--reasoning low|medium|high]
"""

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai/gpt-oss-120b"
CACHE_DIR = "/data/cache/huggingface/hub"
TEST_PATH = Path("math/datasets/combined/test.parquet")
RESULTS_DIR = Path("math/benchmark/results")

SYSTEM_PROMPT = "You are a math problem solver. Solve the problem step by step and put your final answer in \\boxed{}."


def extract_boxed(text: str) -> str:
    match = re.search(r"\\boxed\{(.+?)\}", text)
    return match.group(1).strip() if match else ""


def normalize_answer(ans: str) -> str:
    return ans.strip().replace(" ", "").lower()


def is_correct(predicted: str, expected: str) -> bool:
    if not predicted or not expected:
        return False
    return normalize_answer(predicted) == normalize_answer(expected)


def build_prompt(tokenizer, problem: str, reasoning_effort: str) -> str:
    messages = [{"role": "user", "content": problem}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )


def run(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test set...")
    test_df = pd.read_parquet(TEST_PATH)
    # Drop rows with no ground truth answer — can't evaluate them
    test_df = test_df[test_df["answer"] != ""].reset_index(drop=True)

    if args.limit:
        test_df = test_df.sample(n=args.limit, random_state=42).reset_index(drop=True)

    print(f"Evaluating on {len(test_df):,} problems")
    print(f"Reasoning effort: {args.reasoning}")

    print(f"\nLoading model {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded.")

    predictions = []
    correct_flags = []

    for i in range(0, len(test_df), args.batch_size):
        batch = test_df.iloc[i : i + args.batch_size]

        prompts = [
            build_prompt(tokenizer, row["problem"], args.reasoning)
            for _, row in batch.iterrows()
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            # Extract only the final answer section, not the analysis/thinking trace
            if "assistantfinal" in generated:
                generated = generated.split("assistantfinal", 1)[1]
            predicted = extract_boxed(generated)
            expected = batch.iloc[j]["answer"]
            correct = is_correct(predicted, expected)
            predictions.append(predicted)
            correct_flags.append(correct)

        n_done = min(i + args.batch_size, len(test_df))
        running_acc = sum(correct_flags) / len(correct_flags)
        print(f"  [{n_done}/{len(test_df)}]  running accuracy: {running_acc:.3f}")

    results_df = test_df.copy()
    results_df["predicted"] = predictions
    results_df["correct"] = correct_flags

    out_path = RESULTS_DIR / "baseline.parquet"
    results_df.to_parquet(out_path, index=False)

    # Summary
    total = len(results_df)
    n_correct = sum(correct_flags)
    print(f"\n=== BASELINE RESULTS ===")
    print(f"Overall accuracy: {n_correct}/{total} = {n_correct/total:.3f}")
    print(f"\nPer-source accuracy:")
    for src, grp in results_df.groupby(results_df["source"].str.split("/").str[0]):
        acc = grp["correct"].mean()
        print(f"  {src:<20} {grp['correct'].sum():>5}/{len(grp):<6} = {acc:.3f}")
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on a random subset of N problems")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="medium")
    args = parser.parse_args()
    run(args)
