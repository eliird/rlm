"""
Download a monolingual text dataset for sentence-level LM training.

Uses FineWeb-Edu (sample-10BT subset) — high-quality English web text,
good for segment-level training since documents are long enough to split
into multiple segments at pause boundaries.

Downloads a configurable number of documents and saves as parquet.

Usage:
  python sentence-lm/download_data.py [--num-docs 500000] [--min-length 200]
"""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd


def main(args):
    out_dir = Path("sentence-lm/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "train.parquet"

    if out_path.exists():
        print(f"Already exists: {out_path}")
        return

    print(f"Streaming FineWeb-Edu (sample-10BT), collecting {args.num_docs:,} docs...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    rows = []
    for doc in tqdm(ds, total=args.num_docs, desc="Downloading", unit="doc"):
        text = doc.get("text", "")
        if len(text) < args.min_length:
            continue
        rows.append({"text": text})
        if len(rows) >= args.num_docs:
            break

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} documents to {out_path}")
    print(f"Total characters: {df['text'].str.len().sum():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", type=int, default=500_000,
                        help="Number of documents to download")
    parser.add_argument("--min-length", type=int, default=200,
                        help="Minimum character length per document")
    args = parser.parse_args()
    main(args)
