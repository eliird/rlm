# Hierarchical Sentence-Level Language Model

## Overview

A language model architecture that operates at the sentence level rather than the token level. Instead of predicting the next token, the model predicts the next **sentence embedding**, then decodes that embedding into tokens using a separate autoregressive decoder. This yields significant efficiency gains by reducing the sequence length the core language model must attend over, while preserving output quality through token-level decoding.

---

## Core Idea

Humans don't write by predicting one word at a time in isolation. Given a previous sentence, a writer generally knows the **gist** of what comes next before they spell it out word by word. This architecture mirrors that process:

1. **Think at the sentence level** — predict what the next sentence should mean.
2. **Write at the token level** — realize that meaning as specific words.

---

## Architecture

The system consists of two components with matched hidden dimensions (no projection layers needed).

### Component 1: Segment Encoder (Bidirectional, CLS Token)

- Takes a text segment as input — segments are split at natural pause boundaries (periods, commas, semicolons, colons, etc.) within a 4096-token context window.
- Produces a **fixed-size segment embedding** via a learned **CLS token**.
- Architecture: Bidirectional Transformer encoder. A special CLS token is prepended to the input; the model learns to route all relevant information into that position through full bidirectional self-attention. The CLS hidden state at the final layer is the segment embedding.
- Hidden dimension matches the decoder — CLS vectors slot directly into the decoder's sequence with no projection.
- Can be initialized from a pretrained LLM (decoder-only) by dropping the causal mask to enable bidirectional attention. The model adapts during training.
- **Why CLS over mean pooling:** With JEPA objectives the encoder needs a single target vector to predict. Mean pooling is a passive arithmetic average with no learned compression. The CLS token is explicitly trained to be the prediction target — the entire Transformer learns *what to pack into that one position*. It's a learnable bottleneck rather than a fixed one.

### Component 2: Token-Level Decoder (Pretrained LLM)

- A **pretrained LLM** used as the decoder — no decoder training from scratch. Same hidden dimension as the encoder.
- The decoder operates over a mixed sequence of **CLS embeddings** (compressed past segments) and **raw tokens** (current segment being generated):
  ```
  [CLS₁] [CLS₂] ... [CLSₙ] [tok₁] [tok₂] [tok₃] ...
   ↑ compressed history        ↑ current segment (causal)
  ```
- The decoder **cannot see raw tokens from previous segments** — only their CLS embeddings. This is the key architectural constraint:
  - Forces the encoder to produce high-quality embeddings (the decoder has no way to cheat by looking at old tokens).
  - Eliminates the need for a separate segment-level language model — the decoder itself does sentence-level reasoning through its attention over the CLS sequence.
  - Delivers the efficiency gain: the decoder attends to ~S CLS vectors + current segment tokens, not thousands of historical tokens.
- At each decoding step, the decoder attends to:
  - **CLS embeddings** from all previous segments (one vector per segment, directly in the decoder's hidden space).
  - **Previously generated tokens** in the current segment (causal self-attention).
- No projection layers, no pseudo-tokens — one CLS vector per past segment, straight into the KV cache.

---

## Training

Training is a **single stage** — three losses, all components trained jointly.

### Joint Encoder-Decoder Training (Three Losses, One Stage)

**Goal:** Train the encoder to produce concept-level CLS embeddings that are simultaneously cross-lingually aligned, predictive of what comes next, and decodable. Train the decoder to generate tokens conditioned on CLS history. All in one pass.

**Method:**
- Take a 4096-token context window from any text corpus.
- Split the text at natural pause boundaries: periods, commas, semicolons, colons, etc.
- Each batch is a mix of **parallel multilingual pairs** and **monolingual consecutive segments**.

**Three simultaneous losses:**

1. **Cross-lingual JEPA loss:** Encode a segment in language A → CLS embedding → predictor head → should match EMA-encoder CLS of the same segment in language B. Forces the encoder to learn language-agnostic concept representations — "bonjour" and "hello" must map to the same place.

2. **Next-segment JEPA loss:** Encode segmentₙ → CLS embedding → predictor head → should match EMA-encoder CLS of segmentₙ₊₁. Forces the encoder to capture forward-looking semantic content — the embedding must represent not just what was said, but what it implies comes next.

3. **Reconstruction loss:** The decoder generates segmentₙ₊₁'s tokens conditioned on [CLS₁...CLSₙ] (no raw tokens from prior segments). Cross-entropy on the generated tokens. The decoder *cannot* see prior raw tokens — it must rely on CLS embeddings — so the gradient signal back through the encoder is strong and meaningful. No freezing schedule needed; the architecture itself prevents the decoder from ignoring the embeddings.

**JEPA setup:**
- Context encoder: bidirectional Transformer, produces CLS embedding.
- Target encoder: EMA (exponential moving average) copy of context encoder. Stop-gradient on target — prevents collapse without needing negative pairs.
- Two small predictor heads (lightweight MLP or small Transformer): one for cross-lingual prediction, one for next-segment prediction. They share the same encoder backbone but specialize on their respective tasks.
- Loss: MSE between predictor output and target CLS embedding.

**Per-batch routing:**
- For a multilingual pair: compute losses 1 (cross-lingual JEPA) + 3 (reconstruction).
- For a monolingual consecutive pair: compute losses 2 (next-segment JEPA) + 3 (reconstruction).
- The encoder receives all three gradient signals every step.

**Loss weighting:** The two JEPA losses are both vector-level MSE (similar magnitude), and cross-entropy is the dense per-token signal. In practice it's a two-way balance: combined JEPA weight vs reconstruction weight. Scale the JEPA losses by something proportional to sequence length so gradient magnitudes are comparable, or use a schedule that starts with heavier JEPA weight and gradually introduces more cross-entropy weight.

**Minimum segment length:** Enforce a minimum token count per segment (e.g., 8–16 tokens) to avoid degenerate cases where a segment is just "however," or "in this case," with too little standalone meaning.

**Training data:** Any large text corpus for monolingual segments (infinite training signal). Parallel multilingual corpora (e.g., NLLB-style data) for cross-lingual pairs.

---

## Inference Pipeline

```
Input document (segments s₁ ... sₙ)
        │
        ▼
┌─────────────────────┐
│   Segment Encoder    │  Encode each input segment
│   (bidirectional)    │  into a CLS embedding
└─────────────────────┘
        │
        ▼
   [CLS₁, CLS₂, ..., CLSₙ]   Sequence of CLS embeddings
        │
        ▼
┌─────────────────────┐
│   Decoder (LLM)      │  Attend over CLS history,
│                       │  generate next segment tokens
└─────────────────────┘
        │
        ▼
   "The actual output segment in tokens."
        │
        ▼
┌─────────────────────┐
│   Segment Encoder    │  Encode completed segment
└─────────────────────┘
        │
        ▼
   CLSₙ₊₁ appended to history, raw tokens dropped from KV cache
        │
        ▼
   Continue generating next segment...
```

**Inference loop:**
1. Decoder generates tokens for the current segment, attending to [CLS₁...CLSₙ] + tokens generated so far.
2. Hit segment boundary (EOS / pause symbol).
3. Run encoder over the completed segment → CLSₙ₊₁.
4. Drop raw tokens from KV cache, append CLSₙ₊₁.
5. Continue generating the next segment.

---

## Efficiency Analysis

| Aspect | Standard Token-Level LM | This Architecture |
|---|---|---|
| Decoder context length | All prior tokens (~1000s–10,000s) | ~S CLS vectors + current segment tokens |
| Attention cost per token | O(T) where T = all prior tokens | O(S + t) where S = segments, t = current segment tokens |
| Long-range context | Expensive, requires large context windows | Cheap — one CLS vector per past segment |
| KV cache growth | Linear in total tokens generated | Bounded — raw tokens dropped at each segment boundary, replaced by one CLS entry |
| Components | One model | Encoder + decoder (but no separate segment-level LM) |

The efficiency gain comes from the decoder attending to CLS vectors instead of raw token history. For a 100-segment document with ~20 tokens per segment, the decoder attends to ~100 CLS vectors + ~20 current tokens = ~120 entries, instead of ~2000 raw tokens. The KV cache stays bounded because completed segments are compressed to a single CLS entry.

---

## Open Questions and Risks

### Segment Boundary Design
- Splitting on pause symbols (commas, periods, semicolons, etc.) produces variable-length segments. Some chunks may be very short ("however," or "in this case,") with too little standalone meaning.
- Mitigation: enforce a minimum token count per segment (8–16 tokens). Merge short segments with their neighbors.
- During generation, the decoder produces an EOS token to signal segment completion, or a fixed budget can be imposed.

### Information Bottleneck
- Each past segment is compressed to a single CLS vector. This is aggressive compression — one vector must capture all relevant information from an entire clause or sentence.
- If insufficient: allow the encoder to produce a small set of vectors per segment (e.g., 2–4), trading KV cache size for richer history. But start with one and see how far it goes.

### Reconstruction Fidelity
- The three-loss joint training means the encoder optimizes for cross-lingual alignment, next-segment prediction, and decodability simultaneously. The JEPA losses push toward abstract concepts; the reconstruction loss keeps the embeddings grounded in token-level detail.
- The pretrained decoder provides a strong starting point for fluent generation.
- The decoder's inability to see prior raw tokens provides strong gradient signal — if the CLS embedding is lossy, the decoder's output degrades and the loss increases, forcing the encoder to improve.
- Remaining risk: precise details (numbers, names, rare words) may still be hard to reconstruct from a single compressed vector.

### Accumulation of Error
- At inference time, completed segments are re-encoded through the encoder to produce CLS vectors. Errors in generation propagate through encoding — a badly generated segment produces a bad CLS, which affects all future segments.
- Mitigation: this is inherent to any autoregressive system. The encoder re-encodes actual generated tokens (not predicted embeddings), so there is no embedding-space drift — errors are grounded in real text.

### Cross-Lingual Data Availability
- The cross-lingual JEPA loss requires parallel multilingual corpora. Coverage and quality vary widely across language pairs — low-resource languages may not have enough parallel data to learn good alignment.
- The next-segment JEPA and reconstruction losses work with any monolingual text, so the encoder will still learn useful representations for languages without parallel data — it just won't have explicit cross-lingual alignment for those languages.

---

## Related Work

- **SONAR** (Meta, 2023): Multilingual sentence embedding space with encoder-decoder, trained with translation + auto-encoding + denoising + similarity objectives. Closest existing sentence encoder/decoder infrastructure.
- **Large Concept Model (LCM)** (Meta, 2024): Autoregressive model over SONAR embeddings using MSE/diffusion loss. Directly operates in sentence-embedding space.
- **SONAR-LLM** (2025): Improves on LCM by backpropagating token-level cross-entropy through a frozen SONAR decoder. Best performing sentence-level generation model to date.
- **I-JEPA / V-JEPA** (Meta): JEPA-style self-supervised learning in vision. Inspires the encoder training objective proposed here.
- **Non-Autoregressive Translation** (Gu et al.): Parallel token generation from latent representations. Related efficiency motivation.

---

## Summary

Two components, one training stage, no projection layers. A bidirectional encoder with a learned CLS token compresses each segment into a single vector. A pretrained LLM decoder generates tokens for the current segment, attending only to CLS vectors from past segments — it cannot see prior raw tokens, which forces the encoder to produce high-quality embeddings and eliminates the need for a separate segment-level language model. Three simultaneous losses train the system end-to-end: cross-lingual JEPA (language-agnostic concepts), next-segment JEPA (forward-looking prediction), and reconstruction cross-entropy (decodability). The encoder and decoder share the same hidden dimension — CLS vectors slot directly into the decoder's KV cache with no projection. At each segment boundary, raw tokens are dropped and replaced by a single CLS entry, keeping the KV cache bounded. The result is a model that thinks fast (short CLS sequences) and writes carefully (token-by-token decoding conditioned on compressed context).