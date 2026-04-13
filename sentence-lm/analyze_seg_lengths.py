import argparse
import re
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from tqdm import tqdm

BERT_DIR  = "sentence-lm/bert_weights"
DATA_PATH = "sentence-lm/data/train.parquet"
N_SAMPLE  = 10_000   # default sample size; use --all for full dataset

_PUNCT_BOUNDARY = re.compile(r'(?<=[.?!;:,])\s+')
MIN_SEG_TOKENS  = 8


def split_raw(text: str) -> list[str]:
    """Split on punctuation boundaries without any tokenizer calls."""
    raw = _PUNCT_BOUNDARY.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=N_SAMPLE,
                    help="Number of documents to sample (default 10k)")
parser.add_argument("--all", action="store_true",
                    help="Run on the full dataset")
args = parser.parse_args()

tok = BertTokenizer.from_pretrained(BERT_DIR)
df  = pd.read_parquet(DATA_PATH)
print(f"Documents: {len(df)}")

if not args.all:
    n = min(args.sample, len(df))
    df = df.sample(n, random_state=42).reset_index(drop=True)
    print(f"Sampling {n:,} documents")

# ── Pass 1: collect all raw candidate segments ────────────────────────────────
print("Splitting documents...")
doc_raw_segs = [split_raw(text) for text in tqdm(df["text"])]

# flatten all raw segments for a single batched tokenize call
flat_segs  = [seg for segs in doc_raw_segs for seg in segs]
CHUNK = 100_000
print(f"Tokenizing {len(flat_segs):,} segments in chunks of {CHUNK:,}...")
flat_lens = []
for i in tqdm(range(0, len(flat_segs), CHUNK)):
    flat_lens.extend(tok(flat_segs[i: i + CHUNK], add_special_tokens=False, return_length=True)["length"])

# ── Pass 2: merge short segments using pre-computed lengths ───────────────────
lengths      = []
segs_per_doc = []
ptr = 0
for raw_segs in doc_raw_segs:
    n = len(raw_segs)
    seg_lens = flat_lens[ptr: ptr + n]
    ptr += n

    # greedy merge: accumulate until candidate >= MIN_SEG_TOKENS
    merged_lengths = []
    buf_len = 0
    for l in seg_lens:
        buf_len += l
        if buf_len >= MIN_SEG_TOKENS:
            merged_lengths.append(buf_len)
            buf_len = 0
    if buf_len:
        if merged_lengths:
            merged_lengths[-1] += buf_len
        else:
            merged_lengths.append(buf_len)
    if not merged_lengths:
        merged_lengths = [sum(seg_lens)]

    lengths.extend(merged_lengths)
    segs_per_doc.append(len(merged_lengths))

lengths      = np.array(lengths)
segs_per_doc = np.array(segs_per_doc)

print(f"\n── Segment token lengths ({len(lengths):,} total segments) ──")
for p in [50, 75, 90, 95, 99, 99.9, 100]:
    print(f"  p{p:5.1f}: {np.percentile(lengths, p):6.0f} tokens")
print(f"  mean:  {lengths.mean():6.1f} tokens")

print(f"\n── Segments per document ──")
for p in [50, 75, 90, 95, 99, 100]:
    print(f"  p{p:5.1f}: {np.percentile(segs_per_doc, p):6.0f} segments")
print(f"  mean:  {segs_per_doc.mean():6.1f} segments")

print(f"\n── Suggested config ──")
p95 = int(np.percentile(lengths, 95))
p99 = int(np.percentile(lengths, 99))
p95_segs = int(np.percentile(segs_per_doc, 95))
def ceil8(x): return ((x + 7) // 8) * 8
print(f"  MAX_BERT_LEN / MAX_GPT_LEN (p95): {ceil8(p95)}")
print(f"  MAX_BERT_LEN / MAX_GPT_LEN (p99): {ceil8(p99)}")
print(f"  MAX_SEGMENTS (p95):               {p95_segs}")
