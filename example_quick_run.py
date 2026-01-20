#!/usr/bin/env python3
"""
Quick example run of the LLM-RL training loop.
This runs 3 iterations with 10 episodes each to demonstrate improvement.

Usage:
    .venv/bin/python example_quick_run.py
"""

from main import RLTrainingLoop


def main():
    """Run a quick example training loop."""
    print("\n" + "="*80)
    print("QUICK EXAMPLE RUN - Qwen2-VL-2B-Instruct")
    print("="*80)
    print("This will run 3 iterations with ADAPTIVE episode counts:")
    print("  - Iteration 1: 2 episodes  (quick bootstrap with random policy)")
    print("  - Iteration 2: 4 episodes  (policy improving, gather more data)")
    print("  - Iteration 3: 8 episodes  (refined policy, more episodes for stats)")
    print("Using Qwen2-VL for better vision understanding.")
    print("Each episode runs up to 500 steps (~1-2 mins per episode).")
    print("Total estimated time: 15-30 minutes.")
    print("="*80 + "\n")

    # Create training loop with Qwen2-VL (better vision understanding)
    loop = RLTrainingLoop(
        model_name="Qwen/Qwen2-VL-2B-Instruct",
        experiment_name="quick_test_qwen",
    )

    # Run with settings to see improvement over iterations
    loop.run_training_loop(
        num_iterations=3,            # 3 iterations to see progression
        episodes_per_iteration=16,   # Max episodes (used as cap for adaptive mode)
        adaptive_episodes=True,      # Start small: 2, 4, 8 episodes
        initial_episodes=2,          # Start with just 2 episodes in iteration 1
        max_steps_per_episode=500,   # Longer episodes (typical Pong game)
        reflection_sample_rate=0.3,  # Reflect on 30% of poor actions
        positive_sample_rate=1.0,    # Reflect on ALL positive actions (spatial learning!)
        finetune_epochs=2,           # 2 epochs for fine-tuning
        finetune_batch_size=2,       # Batch size for fine-tuning
        finetune_lr=2e-5,            # Learning rate
        verbose=False,               # Disable verbose to reduce output noise
        frame_skip=4,                # Repeat each action for 4 frames (4x faster!)
    )
    
    # Test Config
    # loop.run_training_loop(
    #     num_iterations=2,            # 2 iterations to see progression
    #     episodes_per_iteration=1,    # 1 episode for quick testing
    #     max_steps_per_episode=20,    # Short episodes for testing
    #     reflection_sample_rate=0.3,  # Reflect on 30% of poor actions
    #     positive_sample_rate=1.0,    # Reflect on ALL successful actions (spatial learning!)
    #     finetune_epochs=2,           # 2 epochs for fine-tuning
    #     finetune_batch_size=2,       # Batch size for fine-tuning
    #     finetune_lr=2e-5,            # Learning rate
    #     verbose=True,                # Enable verbose to see progress
    #     frame_skip=4,                # Repeat each action for 4 frames (4x faster!)
    # )
    

    print("\n" + "="*80)
    print("Quick example complete!")
    print("Check data/experiments/quick_test_qwen/ for results")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
