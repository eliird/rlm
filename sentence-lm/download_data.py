"""
Download and cache FineWeb-Edu (sample-10BT) to the HuggingFace cache.

Run once before training:
  python sentence-lm/download_data.py

The dataset is saved to ~/.cache/huggingface/datasets and loaded from there
during training with memory-mapping (no RAM overhead).
"""

from datasets import load_dataset


def main():
    print("Downloading FineWeb-Edu (sample-10BT) to HF cache...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
    )
    print(f"Done. {len(ds):,} documents cached.")
    print(f"Cache location: {ds.cache_files[0]['filename']}")


if __name__ == "__main__":
    main()
