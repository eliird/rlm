# Math Error-Correction Finetuning on gpt-oss-120b

## Overview

This project builds a targeted error-correction finetuning pipeline on top of **gpt-oss-120b** (OpenAI's open-weight MoE model, 117B total / 5.1B active parameters). The goal is not to build a general math solver, but to teach the model a specific meta-cognitive skill: **recognizing when a familiar reasoning pattern is wrong for a subtly different problem**, and correcting it.

---

## Target Model: gpt-oss-120b

| Property | Detail |
|---|---|
| Architecture | Mixture-of-Experts (MoE) |
| Total Parameters | 117B |
| Active Parameters | 5.1B per token |
| License | Apache 2.0 |
| Context Window | 128K tokens |
| Reasoning | Configurable (low / medium / high) |
| Format | Harmony chat format (required) |
| Hardware | Single H100 80GB GPU |

### Known Weaknesses (Target Areas)
- **Adaptive reasoning**: Scores ~61% on modified/novel problems — confuses pattern recognition for genuine reasoning
- **Inverse scaling**: gpt-oss-20b outperforms it on several benchmarks, suggesting MoE routing inefficiency
- **Verbosity**: Cannot be instructed to be concise — structural characteristic, not prompt-following failure

---

## Core Hypothesis

gpt-oss-120b fails not because it lacks mathematical knowledge, but because it over-relies on **template matching**: it identifies a problem as resembling a familiar type and applies a memorized solution path, even when the problem has a subtle twist that invalidates that approach.

Standard finetuning on correct solutions doesn't fix this. RL with answer-only rewards doesn't fix this. A dataset that explicitly shows the model *where its pattern match went wrong* does.

---

## Pipeline

```
Phase 0 — Dataset Construction (one-time)
  COMBINE primary datasets (EleutherAI/hendrycks_math L4-5, MathInstruct/TheoremQA, NuminaMath-CoT, OpenR1-Math-220k, OpenMathReasoning)
  DEDUPLICATE across sources
  SPLIT → train (80%) / val (10%) / test (10%), stratified by source/difficulty
          ↓
Phase 1 — Baseline Evaluation (one-time)
  RUN base gpt-oss-120b on test set → record accuracy per source/difficulty
  This is the baseline to beat after finetuning
          ↓
Phase 2–4 — Iterative Error Correction (repeat N rounds)

  Each round:
  ┌─────────────────────────────────────────────────────────────┐
  │ 2a. SELECT problems from train split, sampled evenly across │
  │     sources (stratified round-robin)                        │
  │              ↓                                              │
  │ 2b. RUN current model → generates reasoning + answer        │
  │              ↓                                              │
  │ 2c. COMPARE answer to ground truth                          │
  │              ↓                                              │
  │      Correct? → discard (no training signal)                │
  │      Wrong?   → proceed                                     │
  │              ↓                                              │
  │ 2d. GENERATE CORRECTION using current model (high budget):  │
  │     Input:  problem + wrong reasoning (scaffolding only)    │
  │             + reference solution                            │
  │     Output: problem-centric insight:                        │
  │             "Problems like this are often confused with [X] │
  │              because [Y]. Key distinction: [Z].             │
  │              Correct approach: [...]"                       │
  │     Wrong reasoning guides what to address but does NOT     │
  │     appear in the training example.                         │
  │              ↓                                              │
  │ 2e. STOP when target corrections reached (default: 10K)     │
  │              ↓                                              │
  │ 2f. FINETUNE on corrections via SFT (LoRA on attn layers)   │
  │              ↓                                              │
  │ 2g. EVALUATE finetuned model on test set                    │
  │     → compare accuracy to previous round                    │
  │     → check regression on previously correct problems       │
  └─────────────────────────────────────────────────────────────┘
          ↓ repeat with finetuned model as the new base
```

### Why iterative

After each round of finetuning the model changes — easy template-matching failures get fixed, but new blind spots may emerge. Generating all corrections from the base model and finetuning once means by round 2 the data no longer reflects the model's actual failures. Iterating ensures:

- Each round's corrections target the *current* model's mistakes, not a stale snapshot
- Problems that survive each round are progressively harder, providing stronger training signal
- The process naturally terminates when the error rate stops improving

---

## Error Correction Data Format

Training examples are **problem-centric and self-contained** — the wrong attempt is used as scaffolding during generation but does not appear in the final training example.

**Generation prompt (internal, not saved):**
```
Problem:            [math problem]
Common wrong path:  [model's wrong reasoning — used to identify the misconception]
Correct solution:   [reference step-by-step solution]

→ Generate a response that explains this problem as a teacher would,
  addressing the common misconception without referencing a specific student's mistake.
```

**Training example (what gets saved):**
```
User:      [math problem]
Assistant: "Problems like this are often confused with [X] because [Y].
            The key distinction here is [Z].
            Correct approach: [step-by-step reasoning] → [correct answer]"
```

### Why problem-centric framing
- The training example has no missing context — model sees problem, produces insight
- Generalizes better: model learns to anticipate common wrong paths, not react to a specific mistake
- At inference time the model preemptively addresses the likely misconception rather than needing to have already failed

---

## Source Datasets

### Primary — High Error Rate on This Model

| Dataset | Size | Solutions | Why |
|---|---|---|---|
| `EleutherAI/hendrycks_math` (levels 4-5) | ~5K hard problems | ✅ Full step-by-step LaTeX | Hard enough that template matching fails |
| `TIGER-Lab/MathInstruct` (excl. GSM sources) | ~220K | ✅ CoT + PoT | College-level, outside comfort zone |
| `AI-MO/NuminaMath-CoT` | 860K problems | ✅ Full CoT | Large and diverse, full reasoning traces |
| `open-r1/OpenR1-Math-220k` | 220K problems | ✅ 2-4 verified traces per problem | Multiple reasoning paths available |
| `nvidia/OpenMathReasoning` | 306K problems | ✅ CoT + TIR | Pass rates logged — easy to find hard problems |

### Avoid
- `openai/gsm8k` and GSM derivatives (`gsm_train`, `gsm_gpt4`, `gsm_rft`) — model scores ~96%, almost no errors to harvest
- `hendrycks/competition_math` levels 1-3 — too easy, same issue

---

## Training Approach

### Why SFT over RL

| | RL (GRPO) | SFT (this project) |
|---|---|---|
| Signal source | Answer correctness reward | Explicit correction with reasoning |
| Best for | Discovering reasoning strategies | Learning a specific meta-skill |
| Needs teacher model? | No | No — model self-reflects using step-by-step reference solution |
| Data efficiency | Many rollouts per problem | One correction per wrong attempt |
| Outcome ceiling | Can exceed training data quality | Bounded by correction quality |

SFT is the right choice here because you are not asking the model to *discover* how to reason — you are teaching it to **recognize a specific class of failure and apply the correct fix**. That is a pattern learning problem SFT excels at.

> If you want to go further later: add a GRPO stage after SFT using answer correctness as the reward. This is exactly the DeepSeek-R1 approach (SFT cold start → RL). The SFT phase you are building now is the cold-start data.

### DeepSeek-R1 Parallel

DeepSeek found that pure RL (R1-Zero) produced strong reasoning but suffered from repetition, poor readability, and language mixing. Their fix was exactly what you are doing: **SFT cold-start data first, then RL**. Your error-correction dataset maps directly to their Stage 1 cold-start data.

---

## Target Model: DeepSeek-R1-Distill-Qwen-32B

Switched from gpt-oss-120b to DeepSeek-R1-Distill-Qwen-32B. Reasons:
- gpt-oss-120b requires MXFP4 quantization to fit on a single H100, which depends on `triton_kernels` — a package not available in the Kaggle environment
- DeepSeek-R1-Distill-32B fits cleanly in bf16 on 2x H100s (~33GB per GPU), no quantization needed
- Already trained for mathematical reasoning (distilled from R1), strong baseline on competition math

| Property | Detail |
|---|---|
| Architecture | Dense transformer (Qwen2) |
| Parameters | 32B |
| Context Window | 128K tokens |
| Reasoning format | `<think>...</think>` tags in content |
| bf16 VRAM | ~64GB (2x H100) |
| Serving | vLLM, tensor-parallel-size 2 |

---

## Finetuning Notes

- **Full finetune** across all 64 transformer layers — no LoRA
- **Freeze** `embed_tokens` and `lm_head` only; train all attention and MLP layers
- **FSDP** across 8x H100s with `FULL_SHARD` strategy, bf16 mixed precision
- **Adam-Mini** optimizer (`pip install adam-mini`) — ~4-5x less memory than AdamW at this scale; falls back to AdamW if not installed
- **Standard ChatML format** — `<|User|>...<|Assistant|>...` with `<|end▁of▁sentence|>` EOS

---

## Evaluation

Do **not** benchmark on AIME or MATH-500 — the model already scores near-ceiling there.

Instead, build a **held-out error-correction test set**:
1. Collect problems where gpt-oss-120b fails
2. Hold out 10-20% from finetuning
3. Measure whether the finetuned model:
   - Correctly identifies the flaw in the wrong reasoning
   - Produces the right answer
   - Does not regress on problems it already solved correctly

This is the only benchmark that measures the actual target capability.

---

## Key Insight

> gpt-oss-120b already knows the math. What it lacks is the meta-cognitive ability to notice when a familiar problem template is being misapplied. Your error-correction dataset is the only kind of data that directly trains this skill — and it's something that neither standard SFT on correct solutions nor RL with answer-only rewards would teach, because both only reward getting the right answer, not recognizing when a familiar approach is wrong for a subtly different problem.