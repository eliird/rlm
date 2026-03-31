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
Phase 2 — Error Correction Dataset Generation (train + val only)
  1. SELECT problem from train/val split
            ↓
  2. RUN gpt-oss-120b → generates reasoning + answer
            ↓
  3. COMPARE answer to ground truth label
            ↓
       Correct? → discard (no training signal)
       Wrong?   → proceed
            ↓
  4. SELF-REFLECT using gpt-oss-120b (high reasoning budget):
     Input:  problem + model's wrong reasoning + wrong answer + reference solution (step-by-step)
     Output: "You recognized this as [X], but it differs because [Y]. Correct reasoning: [Z]"
     Hypothesis: providing the full step-by-step reference solution gives the model enough
     signal to identify where its reasoning diverged, without needing an external teacher model.
            ↓
  5. FORMAT in Harmony chat format
            ↓
Phase 3 — Finetuning
  6. FINETUNE gpt-oss-120b via SFT (LoRA on attention layers)
```

---

## Error Correction Data Format

Each training example contains **contrastive information** — both the wrong path and the right path — making it more informative per example than standard SFT.

```
Problem:           [math problem]
Model's reasoning: [what the model actually generated]
Model's answer:    [wrong answer]
Correct answer:    [ground truth]
Reference solution:[step-by-step correct solution]

→ Correction output:
"This problem resembles [familiar pattern], which is why you applied [method].
 However, the key difference here is [specific twist].
 The correct approach is: [step-by-step reasoning] → [correct answer]"
```

### Important: Anchor Corrections to the Actual Mistake
The correction generator must reference the **exact wrong reasoning steps** the model produced — not a generic explanation of a common error type. This keeps the dataset genuinely targeted rather than just another CoT dataset.

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

## Finetuning Notes for MoE Architecture

- **Apply LoRA to attention layers**, not MoE feed-forward layers — reasoning patterns live in attention, and inactive experts won't receive gradient updates anyway
- **Harmony format is mandatory** — all training examples must use the Harmony chat template or the model will behave incorrectly
- **Full finetune fits on a single H100 node** per OpenAI's model card, but LoRA/QLoRA is recommended for efficiency
- **Standard SFT frameworks** (Axolotl, LLaMA-Factory) support MoE but require explicit configuration — don't use default settings

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