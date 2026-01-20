#!/usr/bin/env python3
"""
Quick test script to verify gymnasium and Atari environment setup.
Run this to ensure the environment is working before proceeding.

Usage:
    .venv/bin/python test_gymnasium.py
"""

import sys

def test_imports():
    """Test if required libraries can be imported."""
    print("Testing imports...")
    try:
        import gymnasium as gym
        print("✓ gymnasium imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import gymnasium: {e}")
        return False

    try:
        import ale_py
        print("✓ ale_py imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ale_py: {e}")
        return False

    try:
        from PIL import Image
        print("✓ PIL imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PIL: {e}")
        return False

    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False

    return True


def test_pong_environment():
    """Test creating and running Pong environment."""
    print("\nTesting Pong environment...")

    try:
        import gymnasium as gym
        import numpy as np
        from PIL import Image

        # Create environment
        env = gym.make('ALE/Pong-v5')
        print("✓ Created Pong environment")

        # Reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Observation dtype: {obs.dtype}")

        # Convert to PIL Image
        img = Image.fromarray(obs)
        print(f"✓ Converted to PIL Image: {img.size}")

        # Resize to 384x384 (SmolVLM size)
        img_resized = img.resize((384, 384), Image.Resampling.BILINEAR)
        print(f"✓ Resized to SmolVLM format: {img_resized.size}")

        # Test action space
        print(f"  - Action space: {env.action_space}")
        print(f"  - Available actions: {env.unwrapped.get_action_meanings()}")

        # Take a few steps
        for i in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            action_name = env.unwrapped.get_action_meanings()[action]
            print(f"  - Step {i+1}: action={action_name} ({action}), reward={reward}, done={terminated or truncated}")

            if terminated or truncated:
                obs, info = env.reset()
                print("    Episode ended, reset environment")

        env.close()
        print("✓ Environment test complete")
        return True

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Gymnasium + Atari Environment Test")
    print("=" * 60)

    # Test imports
    if not test_imports():
        print("\n❌ Import test failed. Please install dependencies:")
        print("   .venv/bin/pip install 'gymnasium[atari,accept-rom-license]' ale-py pillow numpy")
        sys.exit(1)

    # Test environment
    if not test_pong_environment():
        print("\n❌ Environment test failed.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("✅ All tests passed! Environment is ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
