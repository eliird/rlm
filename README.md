# LLM-Based Reinforcement Learning for Atari Pong

An experimental approach to reinforcement learning that uses a vision-language model (SmolVLM) with a trial-reflect-finetune loop to learn Atari Pong.

## Core Hypothesis

Traditional RL methods require many agents and iterations to learn game-playing strategies. This experiment tests whether a vision-language model with built-in world knowledge can learn more efficiently through:

1. **Trial**: Playing episodes and collecting experience
2. **Reflection**: Using the LLM itself to analyze mistakes and generate corrected actions
3. **Fine-tuning**: Updating model weights on the reflected data

The hypothesis is that because the LLM has physical and spatial understanding, it should learn faster than traditional RL methods.

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING LOOP                         │
│                                                         │
│  ┌──────────┐    ┌────────────┐    ┌──────────────┐   │
│  │  TRIAL   │───>│ REFLECTION │───>│ FINE-TUNING  │   │
│  │  PHASE   │    │   PHASE    │    │    PHASE     │   │
│  └──────────┘    └────────────┘    └──────────────┘   │
│       │               │                    │            │
│       v               v                    v            │
│  Run Episodes   Generate Better     Update Weights     │
│  with Agent     Training Data       → Next Iteration   │
└─────────────────────────────────────────────────────────┘
```

### Model

- **SmolVLM-Instruct (2B parameters)**
  - Vision-language model from HuggingFace
  - Small enough to fine-tune on A100 40GB
  - Fast inference (~3-4x faster than Qwen2-VL)

### Environment

- **Atari Pong** via Gymnasium
- Frames resized to 384x384 (SmolVLM input size)
- Actions: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE

## Project Structure

```
reinforcement_experiments/
├── src/
│   ├── models/
│   │   └── smolvlm_agent.py      # Vision-LLM agent
│   ├── environment/
│   │   └── atari_wrapper.py      # Pong environment wrapper
│   ├── training/
│   │   ├── trial_runner.py       # Episode collection
│   │   ├── reflection.py         # Self-reflection mechanism
│   │   └── finetuner.py          # Full fine-tuning
│   └── utils/
├── configs/
│   └── pong_config.yaml          # Training configuration
├── data/
│   ├── episodes/                 # Collected episodes
│   ├── training_data/            # Reflection-generated data
│   ├── checkpoints/              # Model checkpoints
│   └── experiments/              # Full experiment results
├── main.py                       # Main training loop
├── example_quick_run.py          # Quick test script
├── test_*.py                     # Test scripts
└── requirements.txt
```

## Setup

### 1. Install Dependencies

```bash
# Activate virtual environment
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Test gymnasium and Atari environment
python test_gymnasium.py

# Test SmolVLM agent (downloads ~4GB model)
python test_smolvlm_agent.py

# Test vision understanding
python test_vision_understanding.py
```

## Usage

### Quick Test Run

For a quick test with minimal episodes:

```bash
python example_quick_run.py
```

This runs 2 iterations with 10 episodes each (~15-20 minutes).

### Full Training Run

For a proper experiment:

```bash
python main.py \
  --iterations 5 \
  --episodes 100 \
  --max-steps 1000 \
  --reflection-rate 0.3 \
  --finetune-epochs 3 \
  --finetune-batch-size 4 \
  --finetune-lr 2e-5 \
  --experiment-name "pong_experiment_1"
```

### Command-Line Arguments

- `--iterations`: Number of trial-reflect-finetune cycles (default: 3)
- `--episodes`: Episodes per iteration (default: 20)
- `--max-steps`: Maximum steps per episode (default: 1000)
- `--reflection-rate`: Fraction of poor steps to reflect on (default: 0.3)
- `--finetune-epochs`: Epochs for fine-tuning (default: 3)
- `--finetune-batch-size`: Batch size (default: 4)
- `--finetune-lr`: Learning rate (default: 2e-5)
- `--experiment-name`: Name for this experiment

## How It Works

### Phase 1: Trial (Episode Collection)

```python
agent = SmolVLMAgent(model_name=current_model)
env = PongEnvironment()

# Agent plays Pong, outputting JSON actions
for step in episode:
    action, reasoning, _ = agent.get_action(frame)
    # {"reasoning": "Ball is high, move up", "action": "RIGHT"}

    next_frame, reward, done, _ = env.step(action)
    # Record (frame, action, reasoning, reward, next_frame)
```

### Phase 2: Reflection (Data Generation)

```python
# For steps with poor rewards, ask model to reflect
reflection = agent.get_reflection(
    frame=frame,
    action_taken=action,
    reasoning=reasoning,
    reward=reward,
    next_frame=next_frame
)

# Model outputs:
# {
#   "analysis": "I moved up but ball was low, should have moved down",
#   "correct_action": "LEFT",
#   "correct_reasoning": "Ball is in lower area, need to move paddle down"
# }

# Create training example:
# Input: frame + system prompt
# Output: {"reasoning": "corrected_reasoning", "action": "correct_action"}
```

### Phase 3: Fine-tuning (Weight Update)

```python
# Full fine-tuning on corrected examples
finetuner = SmolVLMFinetuner(model_name=current_model)

checkpoint = finetuner.finetune(
    training_examples=reflection_data,
    num_epochs=3,
    batch_size=4,
    learning_rate=2e-5
)

# Use fine-tuned model for next iteration
current_model = checkpoint
```

## Expected Results

### Baseline (Iteration 1)
- SmolVLM-Instruct has poor Pong understanding
- Mostly copies prompt examples or random actions
- Mean reward: ~ -21 to -15 (loses most games)

### After Fine-tuning (Iterations 2-5)
- Model should learn basic ball tracking
- Better paddle positioning
- Gradual reward improvement

**Success metric**: Mean reward improvement over iterations

## Hardware Requirements

- **GPU**: NVIDIA A100 40GB (or similar)
  - ~5GB for model inference
  - ~20-30GB for fine-tuning
- **Storage**: ~50GB for models and data
- **RAM**: 16GB+ recommended

## Computational Cost

### Per Iteration (100 episodes)
- **Trial phase**: ~30-60 minutes (100 episodes)
- **Reflection phase**: ~15-30 minutes (generating training data)
- **Fine-tuning phase**: ~20-40 minutes (3 epochs)
- **Total**: ~1.5-2.5 hours per iteration

### Full Experiment (5 iterations)
- **Total time**: ~7-12 hours
- **GPU hours**: ~7-12 hours of A100 time

## Monitoring

### TensorBoard

```bash
tensorboard --logdir data/experiments/<experiment_name>/
```

### Check Results

```bash
# View experiment summary
cat data/experiments/<experiment_name>/experiment_summary.json

# View iteration results
cat data/experiments/<experiment_name>/iteration_1/results.json
```

## Troubleshooting

### Out of Memory

- Reduce `--finetune-batch-size` (try 2 or 1)
- Reduce `--episodes` per iteration
- Reduce `--max-steps` per episode

### Slow Training

- Increase `--reflection-rate` to reflect on fewer steps
- Reduce `--finetune-epochs`
- Use fewer `--episodes`

### Poor Performance

- Increase `--episodes` for more training data
- Increase `--reflection-rate` to 0.5 or higher
- Try different `--finetune-lr` (1e-5 or 5e-5)

## Future Improvements

1. **LoRA/QLoRA**: Use parameter-efficient fine-tuning
2. **Curriculum Learning**: Start with simpler tasks
3. **Better Reflection**: Use larger model for reflection
4. **Multi-game**: Extend to other Atari games
5. **Comparison**: Benchmark against DQN, PPO, etc.

## Research Questions

1. How many iterations to reach competent play?
2. Does the model generalize to unseen game states?
3. Is full fine-tuning necessary or is LoRA sufficient?
4. Can we transfer learning to other Atari games?
5. How does this compare to traditional RL in sample efficiency?

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llm_rl_pong,
  title = {LLM-Based Reinforcement Learning for Atari Pong},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/reinforcement_experiments}
}
```

## License

Apache 2.0

## Acknowledgments

- **SmolVLM**: HuggingFace Team
- **Gymnasium**: Farama Foundation
- **ALE**: Arcade Learning Environment team
