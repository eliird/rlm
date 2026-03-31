"""
Download all primary datasets using the HuggingFace CLI.
Run from repo root: .venv/bin/python math/datasets/download.py
"""

import subprocess
import sys
from pathlib import Path

DATASET_DIR = Path(__file__).parent

DATASETS = [
    # (hf_repo, local_name)
    ("EleutherAI/hendrycks_math", "competition_math"),
    ("TIGER-Lab/MathInstruct", "mathinstruct"),
    ("AI-MO/NuminaMath-CoT", "numinamath_cot"),
    ("open-r1/OpenR1-Math-220k", "openr1_math_220k"),
    ("nvidia/OpenMathReasoning", "openmathReasoning"),
]


def download(repo: str, local_name: str) -> None:
    dest = DATASET_DIR / local_name
    cmd = [
        "hf", "download",
        "--repo-type", "dataset",
        "--local-dir", str(dest),
        repo,
    ]

    print(f"\n==> Downloading {repo} → {dest}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"ERROR: failed to download {repo}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    for repo, local_name in DATASETS:
        download(repo, local_name)
    print("\nAll datasets downloaded.")
