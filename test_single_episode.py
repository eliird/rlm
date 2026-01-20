#!/usr/bin/env python3
"""
Test running a single episode with verbose logging.
This helps debug and understand what's happening during gameplay.

Usage:
    .venv/bin/python test_single_episode.py
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment.atari_wrapper import PongEnvironment
from src.models.smolvlm_agent import SmolVLMAgent


def run_single_episode_verbose(max_steps=50):
    """Run one episode with detailed logging."""
    print("\n" + "="*80)
    print("SINGLE EPISODE TEST - VERBOSE MODE")
    print("="*80)
    print(f"Running up to {max_steps} steps\n")

    # Initialize
    print("1. Initializing environment...")
    env = PongEnvironment()

    print("2. Initializing agent (this loads the model)...")
    agent = SmolVLMAgent()

    print("\n3. Starting episode...\n")

    # Reset
    frame, info = env.reset()
    print(f"✓ Environment reset. Frame size: {frame.size}\n")

    total_reward = 0

    for step in range(max_steps):
        print(f"{'─'*80}")
        print(f"STEP {step + 1}/{max_steps}")
        print(f"{'─'*80}")

        # Get action
        print(f"  Getting action from LLM (this may take a few seconds)...")
        start_time = time.time()

        action, reasoning, raw_output = agent.get_action(
            frame,
            temperature=0.7,
        )

        inference_time = time.time() - start_time

        print(f"  ✓ LLM Response ({inference_time:.2f}s):")
        print(f"    Action: {action}")
        print(f"    Reasoning: {reasoning}")
        print(f"    Raw (first 100 chars): {raw_output[:100]}...")

        # Execute
        print(f"\n  Executing action in environment...")
        next_frame, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        print(f"  ✓ Result:")
        print(f"    Reward: {reward}")
        print(f"    Total reward so far: {total_reward}")
        print(f"    Done: {done}")

        if reward != 0:
            print(f"\n  ⚠️  NON-ZERO REWARD! {'+' if reward > 0 else ''}{reward}")

        # Update
        frame = next_frame

        if done:
            print(f"\n  Episode finished!")
            break

        print()  # Blank line between steps

    print("\n" + "="*80)
    print("EPISODE COMPLETE")
    print("="*80)
    print(f"Total steps: {step + 1}")
    print(f"Total reward: {total_reward}")
    print(f"Average inference time: ~{inference_time:.2f}s per step")
    print(f"Estimated time for 100 steps: ~{inference_time * 100 / 60:.1f} minutes")
    print("="*80 + "\n")

    env.close()


if __name__ == "__main__":
    run_single_episode_verbose(max_steps=20)  # Just 20 steps for quick test
