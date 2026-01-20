#!/usr/bin/env python3
"""
Test script for SmolVLM agent with Pong environment.
Tests that the agent can load, process frames, and generate actions.

Usage:
    .venv/bin/python test_smolvlm_agent.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.environment.atari_wrapper import PongEnvironment
from src.models.smolvlm_agent import SmolVLMAgent


def test_agent_initialization():
    """Test agent can be initialized and loaded."""
    print("=" * 60)
    print("Test 1: Agent Initialization")
    print("=" * 60)

    try:
        agent = SmolVLMAgent()
        print("✓ Agent initialized successfully")
        print(f"  Model: {agent.model_name}")
        print(f"  Device: {agent.device}")
        print(f"  Dtype: {agent.dtype}")
        return agent
    except Exception as e:
        print(f"✗ Failed to initialize agent: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_action_generation(agent):
    """Test agent can generate actions from real Pong frames."""
    print("\n" + "=" * 60)
    print("Test 2: Action Generation with Real Pong Frame")
    print("=" * 60)

    try:
        # Create Pong environment and get a real frame
        env = PongEnvironment()
        frame, info = env.reset()

        print(f"✓ Got Pong frame: {frame.size}")

        # Get action from agent
        print("\nGenerating action (this may take a moment)...")
        action, reasoning, raw_output = agent.get_action(
            frame,
            game_context="Game just started, first frame",
            temperature=0.7,
        )

        print(f"\n✓ Action generated successfully")
        print(f"  Action: {action}")
        print(f"  Reasoning: {reasoning}")
        print(f"  Raw output (first 200 chars): {raw_output[:200]}...")

        # Verify action is valid
        valid_actions = env.get_action_space()
        if action in valid_actions:
            print(f"✓ Action '{action}' is valid")
        else:
            print(f"✗ Action '{action}' is NOT in valid actions: {valid_actions}")

        env.close()
        return True

    except Exception as e:
        print(f"✗ Action generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game_interaction(agent):
    """Test agent can play a few steps of Pong."""
    print("\n" + "=" * 60)
    print("Test 3: Multi-Step Game Interaction")
    print("=" * 60)

    try:
        env = PongEnvironment()
        frame, info = env.reset()

        print(f"Playing 5 steps of Pong...")

        total_reward = 0
        for step in range(5):
            # Get action
            action, reasoning, _ = agent.get_action(frame, temperature=0.7)

            # Execute action
            next_frame, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"\nStep {step + 1}:")
            print(f"  Action: {action}")
            print(f"  Reasoning: {reasoning[:80]}...")
            print(f"  Reward: {reward}")
            print(f"  Done: {terminated or truncated}")

            if terminated or truncated:
                print("  Episode ended!")
                break

            frame = next_frame

        print(f"\n✓ Completed game interaction test")
        print(f"  Total reward: {total_reward}")

        env.close()
        return True

    except Exception as e:
        print(f"✗ Game interaction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reflection(agent):
    """Test agent's reflection capability."""
    print("\n" + "=" * 60)
    print("Test 4: Self-Reflection Mechanism")
    print("=" * 60)

    try:
        # Create environment and take one action
        env = PongEnvironment()
        frame, info = env.reset()

        action, reasoning, _ = agent.get_action(frame, temperature=0.7)
        next_frame, reward, _, _, _ = env.step(action)

        print(f"Action taken: {action}")
        print(f"Reasoning: {reasoning}")
        print(f"Reward: {reward}")

        # Get reflection
        print("\nGenerating reflection (this may take a moment)...")
        reflection = agent.get_reflection(
            frame=frame,
            action_taken=action,
            reasoning=reasoning,
            reward=reward,
            next_frame=next_frame,
        )

        print(f"\n✓ Reflection generated successfully")
        print(f"  Analysis: {reflection['analysis'][:100]}...")
        print(f"  Correct action: {reflection['correct_action']}")
        print(f"  Correct reasoning: {reflection['correct_reasoning'][:100]}...")

        env.close()
        return True

    except Exception as e:
        print(f"✗ Reflection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SmolVLM Agent Test Suite")
    print("=" * 60)
    print("\nThis will test:")
    print("1. Agent initialization and model loading")
    print("2. Action generation from Pong frames")
    print("3. Multi-step game interaction")
    print("4. Self-reflection mechanism")
    print("\nNote: This requires transformers, torch, and a GPU for best performance.")
    print("First run may download the SmolVLM model (~4GB).")
    print("=" * 60)

    # Test 1: Initialize agent
    agent = test_agent_initialization()
    if agent is None:
        print("\n❌ Agent initialization failed. Please check dependencies:")
        print("   .venv/bin/pip install torch transformers accelerate pillow")
        sys.exit(1)

    # Test 2: Action generation
    if not test_action_generation(agent):
        print("\n❌ Action generation test failed.")
        sys.exit(1)

    # Test 3: Game interaction
    if not test_game_interaction(agent):
        print("\n❌ Game interaction test failed.")
        sys.exit(1)

    # Test 4: Reflection
    if not test_reflection(agent):
        print("\n❌ Reflection test failed.")
        sys.exit(1)

    # All tests passed
    print("\n" + "=" * 60)
    print("✅ All tests passed! SmolVLM agent is working correctly.")
    print("=" * 60)
    print("\nThe agent is ready to:")
    print("- Process Pong game frames")
    print("- Generate actions with reasoning")
    print("- Reflect on past actions for self-improvement")
    print("\nYou can now proceed with the full training pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    main()
