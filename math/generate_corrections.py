"""
Generate error-correction training data from the train split.

Pipeline per problem:
  1. Run model → get reasoning + answer
  2. Compare answer to ground truth
  3. If wrong → send reflection prompt → model produces full training example:
       <think> fresh reasoning that identifies the pitfall </think>
       clean step-by-step solution ending in \\boxed{}
  4. Save think and response separately as SFT training example

The reflection prompt gives the model the wrong attempt + reference solution as context,
but instructs it to write as if solving fresh — no references to "the correct solution"
or "the wrong approach" in the output.

Problems are sampled evenly across sources. Stops when --target corrections are saved.
Resumable — skips problems already processed.

Requires vLLM server running:
  bash math/benchmark/serve.sh

Run from repo root:
  python math/generate_corrections.py [--target N] [--batch-size N]
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
MODEL_NAME = "deepseek-r1-32b"
TRAIN_PATH = Path("math/datasets/combined/train.parquet")
OUT_DIR = Path("math/data")
CORRECTIONS_PATH = OUT_DIR / "corrections.jsonl"

REFLECTION_PROMPT = """\
Below is a math problem, a flawed attempt, and the correct solution. Use these to inform your response, but do NOT reference them in your output.

Problem:
{problem}

Flawed attempt (shows where reasoning can go wrong):
{wrong_reasoning}

Correct solution (for reference only):
{reference_solution}

Now write a response to the problem as if solving it for the first time, with no knowledge of the above.
In your <think> block, reason carefully — naturally identifying any subtle traps or edge cases in this problem as you work through it.
After </think>, write a clean step-by-step solution ending with the answer in \\boxed{{}}.
Do not mention the flawed attempt or reference solution anywhere in your response.\
"""


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
    ans = ans.strip().lower()
    ans = ans.replace(" ", "")
    ans = ans.replace("\\dfrac", "\\frac")
    ans = ans.replace("\\tfrac", "\\frac")
    ans = ans.replace("\\left(", "(").replace("\\right)", ")")
    ans = ans.replace("\\left[", "[").replace("\\right]", "]")
    return ans


def is_correct(predicted: str, expected: str) -> bool:
    if not predicted or not expected:
        return False
    return normalize(predicted) == normalize(expected)


def chat(messages: list[dict], max_tokens: int = 32768) -> tuple[str, str] | None:
    """
    Returns (think, response) or None on timeout.
    Parses <think>...</think> from content — vLLM strips the opening tag but
    leaves the closing tag, so we strip both ends manually.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=300)
        resp.raise_for_status()
        content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
        # vLLM strips <think> but leaves </think> — split on it
        if "</think>" in content:
            think, response = content.split("</think>", 1)
            think = think.replace("<think>", "").strip()
            response = response.strip()
        else:
            think = ""
            response = content
        return think, response
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


def process(idx: int, row: pd.Series, args: argparse.Namespace) -> dict | str:
    """
    Run one problem through the full pipeline.
    Returns a record dict, or "correct" / "timeout" / "invalid".
    """
    # Step 1: attempt the problem
    result = chat(
        [{"role": "user", "content": row["problem"] + "\n\nSolve the problem step by step. Put your final answer in \\boxed{} at the end."}],
    )
    if result is None:
        return "timeout"
    _, attempt_response = result
    predicted = extract_answer(attempt_response, row["answer"])

    if is_correct(predicted, row["answer"]):
        return "correct"

    # Step 2: reflection — model produces fresh think + solution informed by what went wrong
    prompt = REFLECTION_PROMPT.format(
        problem=row["problem"],
        wrong_reasoning=attempt_response,
        reference_solution=row["solution"],
    )
    result = chat([{"role": "user", "content": prompt}])
    if result is None:
        return "timeout"
    think, response = result
    if not think or not response:
        return "invalid"

    return {
        "idx": int(idx),
        "source": row["source"],
        "problem": row["problem"],
        "correct_answer": row["answer"],
        "model_response": attempt_response,
        "think": think,
        "response": response,
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

    if already_saved >= args.target:
        print(f"Target already reached ({already_saved} >= {args.target}).")
        return

    try:
        requests.get("http://localhost:8000/health", timeout=5).raise_for_status()
    except Exception:
        print("ERROR: vLLM server not reachable. Run: bash math/benchmark/serve.sh")
        return

    out_file = open(CORRECTIONS_PATH, "a")
    stats = {"correct": 0, "corrected": already_saved, "invalid": 0, "timeout": 0}
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
                elif record == "invalid":
                    stats["invalid"] += 1
                else:
                    # Valid record: has think block and boxed answer in response
                    has_box = bool(extract_boxed(record["response"]))
                    if has_box:
                        stats["corrected"] += 1
                        pbar.update(1)
                    else:
                        stats["invalid"] += 1
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()

                pbar.set_postfix({
                    "correct": stats["correct"],
                    "invalid": stats["invalid"],
                    "timeout": stats["timeout"],
                    "in_flight": len(in_flight),
                    "last_s": f"{elapsed:.1f}s",
                })

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
    print(f"  Invalid (skipped)   : {stats['invalid']:,}")
    print(f"\nOutput: {CORRECTIONS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=10_000,
                        help="Stop after this many corrections are saved")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run(args)
