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

The system consists of three components.

### Component 1: Segment Encoder (Bidirectional, CLS Token)

- Takes a text segment as input — segments are split at natural pause boundaries (periods, commas, semicolons, colons, etc.) within a 4096-token context window.
- Produces a **fixed-size segment embedding** via a learned **CLS token**.
- Architecture: Bidirectional Transformer encoder. A special CLS token is prepended to the input; the model learns to route all relevant information into that position through full bidirectional self-attention. The CLS hidden state at the final layer is the segment embedding.
- Can be initialized from a pretrained LLM (decoder-only) by dropping the causal mask to enable bidirectional attention. The model adapts during training.
- **Why CLS over mean pooling:** With JEPA objectives the encoder needs a single target vector to predict. Mean pooling is a passive arithmetic average with no learned compression. The CLS token is explicitly trained to be the prediction target — the entire Transformer learns *what to pack into that one position*. It's a learnable bottleneck rather than a fixed one.

### Component 2: Segment-Level Language Model

- Operates entirely in **segment-embedding space**.
- Takes a sequence of segment embeddings as input.
- Uses self-attention **between segment embeddings** to model document-level structure and flow.
- Predicts the next segment embedding given all previous segment embeddings.
- The sequence length here is the number of segments (typically 10–100), not the number of tokens (typically 1000–10,000+), yielding major efficiency gains.

### Component 3: Token-Level Decoder (Pretrained LLM)

- A **pretrained 1B-parameter LLM** used as the decoder — no decoder training from scratch.
- The segment embedding is projected into a short sequence of **pseudo-tokens** (4–8 vectors via a learned MLP), which are prepended to the decoder's input. This avoids the bottleneck of a single cross-attention target and gives the decoder's attention layers more to work with (similar to prefix tuning / soft prompts).
- At each decoding step, the decoder attends to:
  - **Pseudo-token sequence** projected from the current segment embedding.
  - **Previously generated tokens** in the current segment (causal self-attention).
  - **Previous segment embeddings** from the broader context (cross-attention across segment history).
- This hierarchical attention ensures:
  - Fluent, precise token-level output (pretrained LLM already knows language).
  - Long-range coherence (names, references, terminology from earlier segments are accessible).
  - Efficiency (cross-attending to ~20 segment vectors is far cheaper than attending to thousands of tokens).

---

## Training

Training is divided into two stages.

### Stage 1: Joint Encoder-Decoder Training (Three Losses, One Stage)

**Goal:** Train the encoder to produce concept-level embeddings that are simultaneously cross-lingually aligned, predictive of what comes next, and decodable — all in a single training stage.

**Method:**
- Take a 4096-token context window from any text corpus.
- Split the text at natural pause boundaries: periods, commas, semicolons, colons, etc.
- Each batch is a mix of **parallel multilingual pairs** and **monolingual consecutive segments**.

**Three simultaneous losses on the encoder:**

1. **Cross-lingual JEPA loss:** Encode a segment in language A → CLS embedding → predictor head → should match EMA-encoder CLS of the same segment in language B. Forces the encoder to learn language-agnostic concept representations — "bonjour" and "hello" must map to the same place.

2. **Next-segment JEPA loss:** Encode segmentₙ → CLS embedding → predictor head → should match EMA-encoder CLS of segmentₙ₊₁. Forces the encoder to capture forward-looking semantic content — the embedding must represent not just what was said, but what it implies comes next.

3. **Reconstruction loss:** Encode segmentₙ → CLS embedding → pseudo-token projection → pretrained LLM decoder → cross-entropy on segmentₙ₊₁'s tokens. Ensures the embedding stays decodable and information-rich at the token level.

**JEPA setup:**
- Context encoder: bidirectional Transformer, produces CLS embedding.
- Target encoder: EMA (exponential moving average) copy of context encoder. Stop-gradient on target — prevents collapse without needing negative pairs.
- Two small predictor heads (lightweight MLP or small Transformer): one for cross-lingual prediction, one for next-segment prediction. They share the same encoder backbone but specialize on their respective tasks.
- Loss: MSE between predictor output and target CLS embedding.

**Per-batch routing:**
- For a multilingual pair: compute losses 1 (cross-lingual JEPA) + 3 (reconstruction).
- For a monolingual consecutive pair: compute losses 2 (next-segment JEPA) + 3 (reconstruction).
- The encoder receives all three gradient signals every step.

**Decoder freezing schedule:**
- **Phase A:** Freeze the pretrained decoder. Train only the encoder, predictor heads, and embedding-to-pseudo-token projection. The two JEPA losses don't involve the decoder and run at full strength from step 0. This forces the JEPA objectives to shape the embedding space before reconstruction pressure biases it toward easy-to-decode representations.
- **Phase B:** Unfreeze the decoder and jointly fine-tune all components.

**Loss weighting:** The two JEPA losses are both vector-level MSE (similar magnitude), and cross-entropy is the dense per-token signal. In practice it's a two-way balance: combined JEPA weight vs reconstruction weight. Scale the JEPA losses by something proportional to sequence length so gradient magnitudes are comparable, or use a schedule that starts with heavier JEPA weight and gradually introduces more cross-entropy weight.

**Minimum segment length:** Enforce a minimum token count per segment (e.g., 8–16 tokens) to avoid degenerate cases where a segment is just "however," or "in this case," with too little standalone meaning.

**Training data:** Any large text corpus for monolingual segments (infinite training signal). Parallel multilingual corpora (e.g., NLLB-style data) for cross-lingual pairs.

### Stage 2: Segment-Level Language Model

**Goal:** Train a model that predicts the next segment embedding from a sequence of previous segment embeddings.

**Method:**
- Encode a large document corpus into sequences of segment embeddings using the frozen encoder.
- Train an autoregressive Transformer that operates over these segment-embedding sequences.
- Input: sequence of segment embeddings [s₁, s₂, ..., sₙ].
- Output: predicted embedding for sₙ₊₁.
- Loss options:
  - **MSE loss** in embedding space (simple, but may produce blurry/averaged predictions).
  - **Diffusion-based loss** (as in Meta's LCM — higher quality but more complex).
  - **Token-level cross-entropy through frozen decoder** (as in SONAR-LLM — backpropagate through the decoder to get discrete supervision while still predicting in embedding space).

---

## Inference Pipeline

```
Input document (segments s₁ ... sₙ)
        │
        ▼
┌─────────────────────┐
│   Segment Encoder    │  Encode each input segment
│   (frozen, Stage 1)  │  into a fixed-size embedding
└─────────────────────┘
        │
        ▼
   [e₁, e₂, ..., eₙ]     Sequence of segment embeddings
        │
        ▼
┌─────────────────────┐
│  Segment-Level LM    │  Attend over segment embeddings,
│     (Stage 2)        │  predict next segment embedding
└─────────────────────┘
        │
        ▼
      ê_{n+1}              Predicted next segment embedding
        │
        ▼
┌─────────────────────┐
│  Token-Level Decoder │  Project embedding to pseudo-tokens,
│  (Pretrained LLM)    │  generate tokens autoregressively
└─────────────────────┘
        │
        ▼
   "The actual output segment in tokens."
```

Repeat: encode the generated segment, append its embedding to the sequence, predict the next embedding, decode, and so on.

---

## Efficiency Analysis

| Aspect | Standard Token-Level LM | This Architecture |
|---|---|---|
| Core LM sequence length | Number of tokens (~1000s) | Number of segments (~10s–100s) |
| Attention cost in LM | O(T²) where T = tokens | O(S²) where S = segments |
| Long-range context | Expensive, requires large context windows | Cheap, segments compress context naturally |
| Token generation | Same as core LM (sequential) | Only at final decode step |
| Decoder attention | N/A | Pseudo-token prefix + cross-attention to ~S segment vectors (cheap) |

The main speedup comes from the segment-level LM operating over a much shorter sequence. The token-level decoder still generates sequentially, but only for one segment at a time, conditioned on pseudo-tokens projected from the segment embedding rather than attending to the full token history.

---

## Open Questions and Risks

### Segment Boundary Design
- Splitting on pause symbols (commas, periods, semicolons, etc.) produces variable-length segments. Some chunks may be very short ("however," or "in this case,") with too little standalone meaning.
- Mitigation: enforce a minimum token count per segment (8–16 tokens). Merge short segments with their neighbors.
- During generation, the decoder produces an EOS token to signal segment completion, or a fixed budget can be imposed.

### Information Bottleneck
- A single fixed-size vector is projected to 4–8 pseudo-tokens, which partially addresses this.
- If still insufficient: allow the encoder to produce a small set of vectors per segment rather than exactly one (but this adds complexity to the segment-level LM).

### Reconstruction Fidelity
- The three-loss joint training means the encoder optimizes for cross-lingual alignment, next-segment prediction, and decodability simultaneously. The JEPA losses push toward abstract concepts; the reconstruction loss keeps the embeddings grounded in token-level detail.
- The pretrained decoder provides a strong starting point for fluent generation.
- Remaining risk: precise details (numbers, names, rare words) may still be hard to reconstruct from a compressed embedding. The pseudo-token projection helps but may not fully solve this.

### Accumulation of Error
- At inference time, the segment-level LM predicts embeddings, which are decoded and then re-encoded for the next step. Small errors in predicted embeddings may compound over many segments.
- Mitigation: re-encode the actual generated tokens (not the predicted embedding) before feeding back into the segment LM.

### Quality of Segment-Level Prediction
- MSE loss in embedding space tends to predict the mean of plausible next segments, leading to generic outputs.
- Diffusion or token-level-through-decoder losses may be needed for high-quality generation.

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

The architecture separates **what to say** (segment-level planning) from **how to say it** (token-level realization). A bidirectional encoder with a learned CLS token is trained jointly with a pretrained LLM decoder in a single stage using three simultaneous losses: cross-lingual JEPA (language-agnostic concepts), next-segment JEPA (forward-looking prediction), and reconstruction cross-entropy (decodability). The CLS embedding captures meaning at the concept level — not token-level surface features. A segment-level language model plans the next segment in this embedding space efficiently. The pretrained decoder, conditioned on pseudo-tokens projected from the CLS embedding, generates fluent, coherent output. The result is a model that thinks fast (short segment sequences) and writes carefully (token-by-token decoding with rich context).