#!/usr/bin/env python3
"""
Test disk-based frame storage to verify frames are saved to disk instead of RAM.

Usage:
    .venv/bin/python test_disk_storage.py
"""

import os
import sys
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
from src.training.trial_runner import TrialRunner


def test_disk_storage():
    """Test that frames are saved to disk instead of kept in RAM."""
    print("=" * 70)
    print("Testing Disk-Based Frame Storage")
    print("=" * 70)

    # Initialize
    print("\n1. Initializing environment and agent...")
    env = PongEnvironment()
    agent = SmolVLMAgent(model_name="Qwen/Qwen2-VL-2B-Instruct")

    # Create trial runner with disk storage enabled
    test_dir = "data/test_disk_storage"
    runner = TrialRunner(
        agent=agent,
        env=env,
        save_dir=test_dir,
        save_frames_to_disk=True,  # Enable disk storage
    )

    print(f"\n2. Running 1 episode with 10 steps (saving frames to disk)...")
    episodes = runner.run_trials(
        num_episodes=1,
        max_steps_per_episode=10,
        trial_name="disk_test",
        verbose=False,
    )

    # Check that frames were saved to disk
    print("\n3. Verifying frames were saved to disk...")
    frames_dir = Path(test_dir) / "frames"

    if not frames_dir.exists():
        print(f"❌ ERROR: Frames directory not found at {frames_dir}")
        return False

    # Count frame files
    frame_files = list(frames_dir.glob("*.png"))
    print(f"   Found {len(frame_files)} frame files in {frames_dir}")

    # Check that episode steps contain file paths, not PIL images
    episode = episodes[0]
    first_step = episode.steps[0]

    print(f"\n4. Checking step data format...")
    print(f"   Frame reference type: {type(first_step['frame'])}")
    print(f"   Frame reference: {first_step['frame']}")

    if isinstance(first_step['frame'], str):
        print(f"   ✅ Frame is stored as file path (disk storage working!)")

        # Verify file exists
        if Path(first_step['frame']).exists():
            print(f"   ✅ Frame file exists on disk")
        else:
            print(f"   ❌ Frame file does not exist!")
            return False
    else:
        print(f"   ❌ Frame is stored in memory (disk storage NOT working)")
        return False

    # Calculate approximate memory savings
    print(f"\n5. Memory savings estimation:")
    num_frames = len(frame_files)
    approx_frame_size_mb = 0.5  # ~500KB per PNG frame
    total_saved_mb = num_frames * approx_frame_size_mb
    print(f"   Number of frames: {num_frames}")
    print(f"   Approximate memory saved: {total_saved_mb:.1f} MB")
    print(f"   (vs keeping all frames in RAM)")

    env.close()

    print("\n" + "=" * 70)
    print("✅ Disk storage test PASSED!")
    print("=" * 70)
    print(f"\nFrames saved to: {frames_dir}")
    print(f"Cleanup: rm -rf {test_dir}")
    print()

    return True


if __name__ == "__main__":
    test_disk_storage()