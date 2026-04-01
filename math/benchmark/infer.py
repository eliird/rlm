"""
Inference script: run gpt-oss-120b via vLLM server on the test set and save raw responses.
Resumable — skips problems that already have a saved response.

Requires the vLLM server to be running:
  bash math/benchmark/serve.sh

Run from repo root:
  python math/benchmark/infer.py [--limit N] [--batch-size N] [--reasoning low|medium|high]
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

SERVER_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "gpt-oss-120b"
TEST_PATH = Path("math/datasets/combined/test.parquet")
RESULTS_DIR = Path("math/benchmark/results")
RESPONSES_PATH = RESULTS_DIR / "responses.jsonl"


def load_existing(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            done.add(rec["idx"])
    return done


def query(problem: str, reasoning_effort: str, max_tokens: int = 32768) -> tuple[str, str]:
    """Returns (thinking, final_response)."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": problem + "\n\nSolve the problem step by step. Put your final answer in \\boxed{} at the end."}],
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"reasoning_effort": reasoning_effort},
    }
    resp = requests.post(SERVER_URL, json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    message = data["choices"][0]["message"]
    thinking = message.get("reasoning", "") or ""
    content = message.get("content", "") or ""
    return thinking.strip(), content.strip()


def run(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading test set...")
    test_df = pd.read_parquet(TEST_PATH)
    test_df = test_df[test_df["answer"] != ""].reset_index(drop=True)

    if args.limit:
        test_df = test_df.sample(n=args.limit, random_state=42).reset_index(drop=True)

    done_indices = load_existing(RESPONSES_PATH)
    remaining = test_df[~test_df.index.isin(done_indices)].copy()

    print(f"Total problems : {len(test_df):,}")
    print(f"Already done   : {len(done_indices):,}")
    print(f"Remaining      : {len(remaining):,}")
    print(f"Reasoning      : {args.reasoning}")

    if remaining.empty:
        print("Nothing to do.")
        return

    # Verify server is up
    try:
        requests.get("http://localhost:8000/health", timeout=5).raise_for_status()
    except Exception:
        print("ERROR: vLLM server not reachable at localhost:8000. Run: bash math/benchmark/serve.sh")
        return

    out_file = open(RESPONSES_PATH, "a")
    indices = remaining.index.tolist()

    def fetch(idx_row):
        idx, row = idx_row
        t0 = time.time()
        thinking, response = query(row["problem"], args.reasoning)
        elapsed = time.time() - t0
        return {
            "idx": int(idx),
            "source": row["source"],
            "problem": row["problem"],
            "answer": row["answer"],
            "thinking": thinking,
            "response": response,
            "_elapsed": elapsed,
        }

    # Keep a rolling window of args.batch_size concurrent requests.
    # Uses wait(FIRST_COMPLETED) so whichever request finishes first is
    # processed immediately — the pool stays full at all times.
    with ThreadPoolExecutor(max_workers=args.batch_size) as pool:
        pbar = tqdm(total=len(indices), desc="Generating", unit="problem")
        idx_iter = iter(indices)

        # Seed with first batch
        in_flight = {
            pool.submit(fetch, (idx, test_df.loc[idx]))
            for idx in indices[:args.batch_size]
        }
        idx_iter = iter(indices[args.batch_size:])

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                record = future.result()
                elapsed = record.pop("_elapsed")
                out_file.write(json.dumps(record) + "\n")
                out_file.flush()
                pbar.set_postfix({"last_s": f"{elapsed:.1f}s", "in_flight": len(in_flight)})
                pbar.update(1)

                # Immediately submit next to keep pool full
                next_idx = next(idx_iter, None)
                if next_idx is not None:
                    in_flight.add(pool.submit(fetch, (next_idx, test_df.loc[next_idx])))

        pbar.close()

    out_file.close()
    print(f"\nResponses saved to {RESPONSES_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--reasoning", choices=["low", "medium", "high"], default="medium")
    args = parser.parse_args()
    run(args)
