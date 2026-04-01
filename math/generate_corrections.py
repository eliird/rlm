"""
Generate error-correction training data from the train split.

Pipeline per problem:
  1. Run model → get reasoning + answer
  2. Compare answer to ground truth
  3. If wrong → send reflection prompt → get problem-centric correction
  4. Save as SFT training example: (problem, correction)

Problems are sampled evenly across sources. Stops when --target corrections are saved.
Resumable — skips problems already processed.

Requires vLLM server running:
  bash math/benchmark/serve.sh

Run from repo root:
  python math/generate_corrections.py [--target N] [--batch-size N] [--reasoning-attempt high] [--reasoning-reflect high]
"""

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

SERVER_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"
TRAIN_PATH = Path("math/datasets/combined/train.parquet")
OUT_DIR = Path("math/data")
CORRECTIONS_PATH = OUT_DIR / "corrections.jsonl"

REFLECTION_PROMPT = """\
Below is a math problem, a common incorrect approach students take, and the correct step-by-step solution.

Problem:
{problem}

Common incorrect approach:
{wrong_reasoning}

Correct solution:
{reference_solution}

Explain this problem as an expert teacher would. Do not say "you did X wrong" or reference a specific student's mistake. Instead, explain what makes this problem easy to confuse with a simpler one, what the key distinction is, and walk through the correct reasoning. End with the final answer in \\boxed{{}}.\
"""


def extract_boxed(text: str) -> str:
    match = re.search(r"\\boxed\{(.+?)\}", text)
    return match.group(1).strip() if match else ""


def extract_answer(text: str, expected: str) -> str:
    """
    Try to extract the answer from the model response.
    First tries \boxed{}, then checks if the expected answer appears literally in the response.
    """
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    # Fallback: check if the expected answer appears in the last few lines of the response
    last_lines = " ".join(text.strip().splitlines()[-5:])
    if expected and normalize(expected) in normalize(last_lines):
        return expected
    return ""


def normalize(ans: str) -> str:
    return ans.strip().replace(" ", "").lower()


def is_correct(predicted: str, expected: str) -> bool:
    if not predicted or not expected:
        return False
    return normalize(predicted) == normalize(expected)


def chat(messages: list[dict], reasoning_effort: str, max_tokens: int = 32768) -> tuple[str, str] | None:
    """Returns (thinking, content), or None on timeout."""
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=300)
        resp.raise_for_status()
        msg = resp.json()["choices"][0]["message"]
        return (msg.get("reasoning") or "").strip(), (msg.get("content") or "").strip()
    except requests.exceptions.Timeout:
        return None


def load_done(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            done.add(json.loads(line)["idx"])
    return done


def sample_stratified(train_df: pd.DataFrame, done: set[int]) -> list[int]:
    """
    Return indices sampled evenly across top-level sources, excluding already done.
    Cycles through sources round-robin so the queue is always balanced.
    """
    remaining = train_df[~train_df.index.isin(done)].copy()
    source_col = remaining["source"].str.split("/").str[0]
    groups = {src: grp.index.tolist() for src, grp in remaining.groupby(source_col)}

    # Shuffle each source group
    import random
    rng = random.Random(42)
    for src in groups:
        rng.shuffle(groups[src])

    # Round-robin interleave
    indices = []
    sources = list(groups.keys())
    iters = {src: iter(groups[src]) for src in sources}
    exhausted = set()
    while len(exhausted) < len(sources):
        for src in sources:
            if src in exhausted:
                continue
            idx = next(iters[src], None)
            if idx is None:
                exhausted.add(src)
            else:
                indices.append(idx)

    return indices


def process(idx: int, row: pd.Series, args: argparse.Namespace) -> dict | None:
    """Run one problem through the full pipeline. Returns correction or None if correct."""
    result = chat(
        [{"role": "user", "content": row["problem"] + "\n\nSolve the problem step by step. Put your final answer in \\boxed{} at the end."}],
        reasoning_effort=args.reasoning_attempt,
    )
    if result is None:
        return "timeout"
    _, response = result
    predicted = extract_answer(response, row["answer"])

    if is_correct(predicted, row["answer"]):
        return "correct"

    prompt = REFLECTION_PROMPT.format(
        problem=row["problem"],
        wrong_reasoning=response,
        reference_solution=row["solution"],
    )
    result = chat(
        [{"role": "user", "content": prompt}],
        reasoning_effort=args.reasoning_reflect,
    )
    if result is None:
        return "timeout"
    _, correction = result

    return {
        "idx": int(idx),
        "source": row["source"],
        "problem": row["problem"],
        "correct_answer": row["answer"],
        "correct_solution": row["solution"],
        "model_response": response,
        "correction": correction,
    }


def run(args: argparse.Namespace):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train set...")
    train_df = pd.read_parquet(TRAIN_PATH)
    train_df = train_df[train_df["answer"] != ""].reset_index(drop=True)

    done = load_done(CORRECTIONS_PATH)
    already_saved = len(done)
    indices = sample_stratified(train_df, done)

    print(f"Train problems  : {len(train_df):,}")
    print(f"Already saved   : {already_saved:,}")
    print(f"Target          : {args.target:,}")
    print(f"Remaining to hit target (est): {max(0, args.target - already_saved):,} corrections needed")
    print(f"Attempt budget  : {args.reasoning_attempt}")
    print(f"Reflect budget  : {args.reasoning_reflect}")

    if already_saved >= args.target:
        print(f"Target already reached ({already_saved} >= {args.target}).")
        return

    try:
        requests.get("http://localhost:8000/health", timeout=5).raise_for_status()
    except Exception:
        print("ERROR: vLLM server not reachable. Run: bash math/benchmark/serve.sh")
        return

    out_file = open(CORRECTIONS_PATH, "a")
    stats = {"correct": 0, "corrected": already_saved, "empty": 0, "timeout": 0}
    stop = False

    def fetch(idx_row):
        idx, row = idx_row
        t0 = time.time()
        result = process(idx, row, args)
        return idx, result, time.time() - t0

    with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
        pbar = tqdm(total=args.target, initial=already_saved, desc="Corrections", unit="correction")

        in_flight = {
            pool.submit(fetch, (idx, train_df.loc[idx]))
            for idx in indices[:args.batch_size]
        }
        idx_iter = iter(indices[args.batch_size:])

        while in_flight:
            done_futures, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done_futures:
                idx, record, elapsed = future.result()

                if record == "correct":
                    stats["correct"] += 1
                elif record == "timeout":
                    stats["timeout"] += 1
                else:
                    has_box = bool(extract_boxed(record["correction"]))
                    if has_box:
                        stats["corrected"] += 1
                    else:
                        stats["empty"] += 1
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                    if has_box:
                        pbar.update(1)

                pbar.set_postfix({
                    "correct": stats["correct"],
                    "no_box": stats["empty"],
                    "timeout": stats["timeout"],
                    "in_flight": len(in_flight),
                    "last_s": f"{elapsed:.1f}s",
                })

                # Stop submitting new work once target is reached
                if stats["corrected"] >= args.target:
                    stop = True


                if not stop:
                    next_idx = next(idx_iter, None)
                    if next_idx is not None:
                        in_flight.add(pool.submit(fetch, (next_idx, train_df.loc[next_idx])))

        pbar.close()

    out_file.close()
    print(f"\nDone.")
    print(f"  Corrections saved   : {stats['corrected']:,}  (target: {args.target:,})")
    print(f"  Correct (discarded) : {stats['correct']:,}")
    print(f"  Timed out (skipped) : {stats['timeout']:,}")
    print(f"  No boxed answer     : {stats['empty']:,}")
    print(f"\nOutput: {CORRECTIONS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10_000,
                        help="Stop after this many corrections are saved")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reasoning-attempt", choices=["low", "medium", "high"], default="high")
    parser.add_argument("--reasoning-reflect", choices=["low", "medium", "high"], default="high")
    args = parser.parse_args()
    run(args)
