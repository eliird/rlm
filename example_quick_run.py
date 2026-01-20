#!/usr/bin/env python3
"""
Quick example run of the LLM-RL training loop.
This is a minimal example with just 2 iterations and 10 episodes each.

Usage:
    .venv/bin/python example_quick_run.py
"""

from main import RLTrainingLoop


def main():
    """Run a quick example training loop."""
    print("\n" + "="*80)
    print("QUICK EXAMPLE RUN - Qwen2-VL-2B-Instruct")
    print("="*80)
    print("This will run 2 iterations with 5 episodes each.")
    print("Using Qwen2-VL for better vision understanding.")
    print("This is just for testing - for real training use more episodes.")
    print("="*80 + "\n")

    # Create training loop with Qwen2-VL (better vision understanding)
    loop = RLTrainingLoop(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        experiment_name="quick_test_qwen",
    )

    # Run with minimal settings for quick testing
    loop.run_training_loop(
        num_iterations=1,           # Just 2 iterations
        episodes_per_iteration=1,   # Only 5 episodes per iteration (faster testing)
        max_steps_per_episode=20,  # Much shorter episodes for testing
        reflection_sample_rate=0.5, # Sample more for small dataset
        finetune_epochs=1,          # Just 1 epoch for testing
        finetune_batch_size=2,      # Smaller batch
        finetune_lr=2e-5,
        verbose=True,               # Enable verbose logging to see progress
    )

    print("\n" + "="*80)
    print("Quick example complete!")
    print("Check data/experiments/quick_test_qwen/ for results")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
