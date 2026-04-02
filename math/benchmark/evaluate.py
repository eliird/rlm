"""
Evaluation script: score saved responses from infer.py.

Reads:  math/benchmark/results/responses.jsonl
Writes: math/benchmark/results/eval.parquet
        math/benchmark/results/eval_summary.txt

Run from repo root:
  python math/benchmark/evaluate.py
"""

import re
from pathlib import Path

import pandas as pd

import argparse


def extract_boxed(text: str) -> str:
    """Extract content from \\boxed{}, handling nested braces."""
    idx = text.find(r"\boxed{")
    if idx == -1:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i - 1].strip()


def normalize(ans: str) -> str:
    ans = ans.strip().lower()
    ans = ans.replace(" ", "")
    # Treat display/text variants as equivalent
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    return ans


def is_correct(predicted: str, expected: str) -> bool:
    if not predicted or not expected:
        return False
    return normalize(predicted) == normalize(expected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="math/benchmark/results/responses.jsonl",
                        help="Responses JSONL from infer.py")
    parser.add_argument("--output-dir", type=str, default="math/benchmark/results",
                        help="Directory to write eval.parquet and eval_summary.txt")
    args = parser.parse_args()

    responses_path = Path(args.input)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not responses_path.exists():
        print(f"No responses found at {responses_path}. Run infer.py first.")
        return

    print(f"Loading responses from {responses_path}...")
    df = pd.read_json(responses_path, lines=True)
    print(f"  {len(df):,} responses loaded")

    df["predicted"] = df["response"].apply(extract_boxed)
    df["correct"] = df.apply(lambda r: is_correct(r["predicted"], r["answer"]), axis=1)

    total = len(df)
    n_correct = df["correct"].sum()
    overall_acc = n_correct / total

    lines = []
    lines.append(f"=== BASELINE EVALUATION ===")
    lines.append(f"Total problems : {total:,}")
    lines.append(f"Correct        : {n_correct:,}")
    lines.append(f"Overall accuracy: {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    lines.append("")
    lines.append("Per-source accuracy:")

    source_col = df["source"].str.split("/").str[0]
    for src, grp in df.groupby(source_col):
        acc = grp["correct"].mean()
        lines.append(f"  {src:<22} {grp['correct'].sum():>6}/{len(grp):<7} {acc*100:.2f}%")

    lines.append("")
    lines.append("Problems with no boxed answer extracted:")
    no_answer = df[df["predicted"] == ""]
    lines.append(f"  {len(no_answer):,} / {total:,}  ({len(no_answer)/total*100:.1f}%)")

    summary = "\n".join(lines)
    print("\n" + summary)

    summary_path = results_dir / "eval_summary.txt"
    summary_path.write_text(summary)
    print(f"\nSummary saved to {summary_path}")

    eval_path = results_dir / "eval.parquet"
    df.to_parquet(eval_path, index=False)
    print(f"Full results saved to {eval_path}")


if __name__ == "__main__":
    main()
