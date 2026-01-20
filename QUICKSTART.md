# Quick Start Guide

## Installation (5 minutes)

```bash
# Install dependencies
.venv/bin/pip install -r requirements.txt
```

## Verify Setup (10 minutes)

```bash
# 1. Test Atari environment
.venv/bin/python test_gymnasium.py
# Expected: ✅ All tests passed!

# 2. Test SmolVLM agent (downloads ~4GB model first time)
.venv/bin/python test_smolvlm_agent.py
# Expected: ✅ All tests passed! (model generates actions)

# 3. Test vision understanding (optional)
.venv/bin/python test_vision_understanding.py
# This shows what the model "sees" in Pong frames
```

## Run Quick Test (15-20 minutes)

```bash
# Run 2 iterations with 10 episodes each
.venv/bin/python example_quick_run.py
```

This will:
1. **Iteration 1**: Run 10 episodes with base model → Reflect → Fine-tune
2. **Iteration 2**: Run 10 episodes with fine-tuned model → Reflect → Fine-tune
3. Save all results to `data/experiments/quick_test/`

## Run Full Experiment (7-12 hours)

```bash
# Run proper training with 5 iterations, 100 episodes each
.venv/bin/python main.py \
  --iterations 5 \
  --episodes 100 \
  --experiment-name "pong_full_experiment"
```

## Check Results

```bash
# View summary
cat data/experiments/quick_test/experiment_summary.json

# View reward progression
grep "mean_reward" data/experiments/quick_test/*/results.json
```

## Understanding the Output

### Directory Structure After Running
```
data/experiments/quick_test/
├── experiment_summary.json          # Overall results
├── iteration_1/
│   ├── results.json                 # Iteration metrics
│   ├── episodes/
│   │   └── iteration_1/
│   │       └── metadata.json        # Episode data
│   ├── training_data/
│   │   └── iteration_1.json         # Reflection-generated training data
│   └── checkpoints/
│       └── iteration_1/
│           └── final/               # Fine-tuned model
└── iteration_2/
    └── ...
```

### Key Metrics

**experiment_summary.json**:
```json
{
  "experiment_name": "quick_test",
  "num_iterations": 2,
  "final_model": "path/to/final/checkpoint",
  "iteration_history": [
    {
      "iteration": 1,
      "trial_statistics": {
        "mean_reward": -18.5,    // ← Track this!
        "num_episodes": 10
      },
      "num_training_examples": 45
    },
    {
      "iteration": 2,
      "trial_statistics": {
        "mean_reward": -15.2,    // ← Should improve!
        "num_episodes": 10
      },
      "num_training_examples": 38
    }
  ]
}
```

## Expected Behavior

### Baseline (Iteration 1)
- Mean reward: **-21 to -15** (model loses most points)
- Model mostly copies prompt examples
- Poor ball tracking

### After Fine-tuning (Iteration 2+)
- Mean reward: **Should improve gradually**
- Better action diversity
- Some ball tracking ability

### Success Criteria
✅ **Reward improves across iterations**
✅ **Model generates diverse actions (not just repeating examples)**
✅ **Reflection generates meaningful corrections**

## Troubleshooting

### "Out of Memory" Error
```bash
# Reduce batch size
.venv/bin/python main.py --finetune-batch-size 2 --episodes 20
```

### "Too Slow"
```bash
# Reduce episodes and epochs
.venv/bin/python main.py --iterations 3 --episodes 50 --finetune-epochs 2
```

### "Permission Error" for HuggingFace Cache
The code automatically uses `data/hf_cache/` in your project directory. If you still see errors, manually set:
```bash
export HF_HOME=/data1/work/irdali.durrani/reinforcement_experiments/data/hf_cache
```

## Next Steps

1. ✅ **Run quick test** to verify everything works
2. 📊 **Check if reward improves** between iterations
3. 🚀 **Run full experiment** (5 iterations, 100 episodes)
4. 📈 **Analyze results** and compare to baseline
5. 🔬 **Experiment** with hyperparameters

## Customization

### Change Model
```python
# In main.py or example_quick_run.py
loop = RLTrainingLoop(
    model_name="path/to/your/model",  # Any vision-LLM
    experiment_name="custom_experiment"
)
```

### Adjust Hyperparameters
```bash
python main.py \
  --iterations 10 \
  --episodes 200 \
  --finetune-lr 1e-5 \
  --reflection-rate 0.5
```

## Questions?

Check [README.md](README.md) for detailed documentation.

## Important Notes

⚠️ **First run downloads SmolVLM (~4GB)** - be patient!
⚠️ **GPU required** - needs CUDA-capable GPU
⚠️ **Disk space** - Full experiment needs ~20-50GB
⏰ **Time** - Full experiment takes 7-12 hours on A100

Enjoy experimenting! 🎮🤖
