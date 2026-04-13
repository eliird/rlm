import re
import numpy as np
import pandas as pd
from transformers import BertTokenizer

BERT_DIR  = "sentence-lm/bert_weights"
DATA_PATH = "sentence-lm/data/train.parquet"

_PUNCT_BOUNDARY = re.compile(r'(?<=[.?!;:,])\s+')
MIN_SEG_TOKENS  = 8


def split_into_segments(text: str) -> list[str]:
    raw = _PUNCT_BOUNDARY.split(text.strip())
    raw = [s.strip() for s in raw if s.strip()]

    merged, buffer = [], ""
    for seg in raw:
        candidate = (buffer + " " + seg).strip() if buffer else seg
        if len(tok.tokenize(candidate)) >= MIN_SEG_TOKENS:
            merged.append(candidate)
            buffer = ""
        else:
            buffer = candidate

    if buffer:
        if merged:
            merged[-1] = merged[-1] + " " + buffer
        else:
            merged.append(buffer)

    return merged if merged else [text.strip()]


tok = BertTokenizer.from_pretrained(BERT_DIR)
df  = pd.read_parquet(DATA_PATH)
print(f"Documents: {len(df)}")

lengths      = []
segs_per_doc = []

for text in df["text"]:
    segs = split_into_segments(text)
    segs_per_doc.append(len(segs))
    for seg in segs:
        lengths.append(len(tok.tokenize(seg)))

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
# round up to nearest multiple of 8 for memory alignment
def ceil8(x): return ((x + 7) // 8) * 8
print(f"  MAX_BERT_LEN / MAX_GPT_LEN (p95): {ceil8(p95)}")
print(f"  MAX_BERT_LEN / MAX_GPT_LEN (p99): {ceil8(p99)}")
print(f"  MAX_SEGMENTS (p95):               {p95_segs}")
