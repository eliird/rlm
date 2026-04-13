"""
Download a monolingual text dataset for sentence-level LM training.

Uses FineWeb-Edu (sample-10BT subset) — high-quality English web text,
good for segment-level training since documents are long enough to split
into multiple segments at pause boundaries.

Downloads a configurable number of documents and saves as parquet.

Usage:
  python sentence-lm/download_data.py                      # 500k docs (default)
  python sentence-lm/download_data.py --num-docs 1000000   # 1M docs
  python sentence-lm/download_data.py --all                # full dataset (~9.6M docs)
  python sentence-lm/download_data.py --overwrite          # re-download even if file exists
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

    if out_path.exists() and not args.overwrite:
        print(f"Already exists: {out_path}  (use --overwrite to re-download)")
        return

    limit = None if args.all else args.num_docs
    desc = f"Downloading {'all' if args.all else f'{limit:,}'} docs"
    print(f"Streaming FineWeb-Edu (sample-10BT)... {desc}")

    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    rows = []
    for doc in tqdm(ds, total=limit, desc="Downloading", unit="doc"):
        text = doc.get("text", "")
        if len(text) < args.min_length:
            continue
        rows.append({"text": text})
        if limit is not None and len(rows) >= limit:
            break

    df = pd.DataFrame(rows)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df):,} documents to {out_path}")
    print(f"Total characters: {df['text'].str.len().sum():,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-docs", type=int, default=500_000,
                        help="Number of documents to download (default 500k, ignored if --all)")
    parser.add_argument("--all", action="store_true",
                        help="Download the entire dataset (~9.6M docs)")
    parser.add_argument("--min-length", type=int, default=200,
                        help="Minimum character length per document")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing file")
    args = parser.parse_args()
    main(args)
