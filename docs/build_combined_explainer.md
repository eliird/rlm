# build_combined.py — What it does

Script: `math/datasets/build_combined.py`
Output: `math/datasets/combined/train.parquet` and `math/datasets/combined/test.parquet`

All four datasets are normalized to the same schema:

| Column | Type | Description |
|---|---|---|
| `problem` | str | The math problem text |
| `solution` | str | Full step-by-step solution |
| `answer` | str | Final answer extracted from `\boxed{}` in the solution, empty string if not present |
| `source` | str | Dataset and sub-source identifier (e.g. `competition_math/algebra`) |

---

## Per-dataset processing

### competition_math (`EleutherAI/hendrycks_math`)

- Source files are organized by category folder (algebra, geometry, etc.), each with a `train` and `test` parquet file.
- **Filter**: keep only `Level 4` and `Level 5` problems — levels 1-3 are too easy for the model to make errors on.
- **Split**: uses the existing train/test files directly, preserving the original split per category.
- Columns used: `problem`, `solution`. Answer extracted from `\boxed{}` in solution.

### MathInstruct (`TIGER-Lab/MathInstruct`)

- Source is a single large JSON file (~262K rows) with problems from many sub-sources.
- **Filter**: drops all GSM8K derivatives (`gsm_train`, `gsm_gpt4`, `gsm_rft`) — the model already solves these at ~96% accuracy, so they produce almost no errors to train on. Remaining ~220K rows kept.
- **Split**: no pre-existing split — randomly split 90% train / 10% test (seed 42).
- Columns: `instruction` → `problem`, `output` → `solution`. Answer extracted from `\boxed{}` where present (many solutions don't use LaTeX boxed format, so `answer` will be empty for those).

### NuminaMath-CoT (`AI-MO/NuminaMath-CoT`)

- Source is 5 train parquet shards + 1 test parquet shard.
- **Split**: the original split has 860K train / 100 test which is unusable. All shards are combined and randomly split 90% train / 10% test (seed 42).
- Columns used: `problem`, `solution`, `source` (sub-source tag like `synthetic_math`, `amc_aime`, etc., prefixed with `numinamath/`).
- Answer extracted from `\boxed{}` in solution.

### OpenR1-Math-220k (`open-r1/OpenR1-Math-220k`)

- The dataset has three subsets: `data`, `extended`, and `all`. Only the `data` subset is used — it contains the clean, correctness-verified problems. `extended` adds more generations but with lower verification confidence.
- **Split**: no pre-existing test split — randomly split 90% train / 10% test (seed 42).
- Columns used: `problem`, `solution`, `answer` (already present in this dataset), `source`.

---

## Final step

After all four datasets are loaded and normalized, the train rows are concatenated and shuffled, and the test rows are concatenated and shuffled. Both are written to parquet.

The test set is locked from this point — it is not used during error-correction dataset generation or finetuning, only for baseline evaluation and post-finetune comparison.
