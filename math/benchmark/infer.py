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
MODEL_NAME = "deepseek-r1-32b"
TEST_PATH = Path("math/datasets/combined/test.parquet")


def load_existing(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            done.add(rec["idx"])
    return done





def query(problem: str, max_tokens: int = 32768) -> tuple[str, str] | None:
    """Returns (thinking, final_response), or None on timeout."""
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": problem + "\n\nSolve the problem step by step. Put your final answer in \\boxed{} at the end."}],
        "max_tokens": max_tokens,
    }
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=600)
        resp.raise_for_status()
    except requests.exceptions.Timeout:
        return None
    content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
    if "</think>" in content:
        thinking, response = content.split("</think>", 1)
        thinking = thinking.replace("<think>", "").strip()
        response = response.strip()
    else:
        thinking, response = "", content
    return thinking, response


def run(args):
    responses_path = Path(args.output)
    responses_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading test set...")
    test_df = pd.read_parquet(TEST_PATH)
    test_df = test_df[test_df["answer"] != ""].reset_index(drop=True)

    if args.limit:
        test_df = test_df.sample(n=args.limit, random_state=42).reset_index(drop=True)

    done_indices = load_existing(responses_path)
    remaining = test_df[~test_df.index.isin(done_indices)].copy()

    print(f"Total problems : {len(test_df):,}")
    print(f"Already done   : {len(done_indices):,}")
    print(f"Remaining      : {len(remaining):,}")

    if remaining.empty:
        print("Nothing to do.")
        return

    # Verify server is up
    try:
        requests.get("http://localhost:8000/health", timeout=5).raise_for_status()
    except Exception:
        print("ERROR: vLLM server not reachable at localhost:8000. Run: bash math/benchmark/serve.sh")
        return

    out_file = open(responses_path, "a")
    indices = remaining.index.tolist()

    def fetch(idx_row):
        idx, row = idx_row
        t0 = time.time()
        result = query(row["problem"])
        elapsed = time.time() - t0
        if result is None:
            return None, elapsed
        thinking, response = result
        return {
            "idx": int(idx),
            "source": row["source"],
            "problem": row["problem"],
            "answer": row["answer"],
            "thinking": thinking,
            "response": response,
        }, elapsed

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
                record, elapsed = future.result()
                if record is not None:
                    out_file.write(json.dumps(record) + "\n")
                    out_file.flush()
                pbar.set_postfix({"last_s": f"{elapsed:.1f}s", "in_flight": len(in_flight), "timeout": record is None})
                pbar.update(1)

                # Immediately submit next to keep pool full
                next_idx = next(idx_iter, None)
                if next_idx is not None:
                    in_flight.add(pool.submit(fetch, (next_idx, test_df.loc[next_idx])))

        pbar.close()

    out_file.close()
    print(f"\nResponses saved to {responses_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output", type=str, default="math/benchmark/results/responses.jsonl",
                        help="Path to write responses JSONL")
    args = parser.parse_args()
    run(args)
