#!/usr/bin/env python3
"""
Test the reflection mechanism in isolation.
This will show us what the model outputs when asked to reflect.

Usage:
    .venv/bin/python test_reflection.py
    .venv/bin/python test_reflection.py --model HuggingFaceTB/SmolVLM-Instruct
"""

import os
import sys
import argparse
from pathlib import Path

# Set HuggingFace cache BEFORE any imports
_project_root = Path(__file__).parent
_cache_dir = str(_project_root / 'data' / 'hf_cache')
os.environ['HF_HOME'] = _cache_dir
os.environ['TRANSFORMERS_CACHE'] = _cache_dir
os.environ['HF_HUB_CACHE'] = _cache_dir

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment.atari_wrapper import PongEnvironment
from src.models.smolvlm_agent import SmolVLMAgent


def test_reflection(model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
    """Test reflection mechanism."""
    print("=" * 70)
    print(f"Testing Reflection Mechanism")
    print(f"Model: {model_name}")
    print("=" * 70)

    # Initialize
    print("\n1. Initializing environment and agent...")
    env = PongEnvironment()
    agent = SmolVLMAgent(model_name=model_name)

    # Get two frames (before and after an action)
    print("\n2. Getting game frames...")
    frame, _ = env.reset()

    # Take an action
    action = "RIGHT"
    reasoning = "Ball is high, moving paddle up"
    next_frame, reward, _, _, _ = env.step(action)

    print(f"   Action taken: {action}")
    print(f"   Reasoning: {reasoning}")
    print(f"   Reward: {reward}")

    # Test reflection
    print("\n3. Generating reflection...")
    print("   (This may take a few seconds...)")
    print()

    reflection = agent.get_reflection(
        frame=frame,
        action_taken=action,
        reasoning=reasoning,
        reward=reward,
        next_frame=next_frame,
    )

    print("\n" + "=" * 70)
    print("REFLECTION RESULT")
    print("=" * 70)
    print(f"Analysis: {reflection['analysis']}")
    print()
    print(f"Correct Action: {reflection['correct_action']}")
    print()
    print(f"Correct Reasoning: {reflection['correct_reasoning']}")
    print("=" * 70)

    # Validate
    valid_actions = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    if reflection['correct_action'] in valid_actions:
        print("-------------------")
        print(f"\n✅ Valid action generated: {reflection['correct_action']}")
        for k, v in reflection:
            print(f"{k} : {v}")
    else:
        print(f"\n❌ Invalid action: {reflection['correct_action']}")
        print(f"   Valid actions are: {', '.join(valid_actions)}")

    env.close()

    return reflection


def test_multiple_reflections(model_name: str = "Qwen/Qwen2-VL-2B-Instruct", num_tests: int = 5):
    """Test multiple reflections to see consistency."""
    print("\n" + "=" * 70)
    print(f"Testing {num_tests} reflections for consistency")
    print("=" * 70)

    env = PongEnvironment()
    agent = SmolVLMAgent(model_name=model_name)

    actions_generated = []

    for i in range(num_tests):
        print(f"\nTest {i+1}/{num_tests}...")

        frame, _ = env.reset()
        next_frame, reward, _, _, _ = env.step("RIGHT")

        reflection = agent.get_reflection(
            frame=frame,
            action_taken="RIGHT",
            reasoning="Moving paddle up",
            reward=reward,
            next_frame=next_frame,
        )

        print(f"  Correct action: {reflection['correct_action']}")
        actions_generated.append(reflection['correct_action'])

    env.close()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Actions generated: {actions_generated}")

    valid_actions = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
    valid_count = sum(1 for a in actions_generated if a in valid_actions)

    print(f"Valid: {valid_count}/{num_tests}")
    print(f"Success rate: {valid_count/num_tests*100:.1f}%")

    if valid_count == num_tests:
        print("\n✅ All reflections generated valid actions!")
    else:
        print(f"\n⚠️  Only {valid_count}/{num_tests} reflections were valid")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test reflection mechanism")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Test multiple reflections"
    )
    parser.add_argument(
        "--num",
        type=int,
        default=5,
        help="Number of reflections to test (with --multiple)"
    )

    args = parser.parse_args()

    if args.multiple:
        test_multiple_reflections(model_name=args.model, num_tests=args.num)
    else:
        test_reflection(model_name=args.model)