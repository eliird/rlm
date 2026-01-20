"""
Atari Pong environment wrapper for SmolVLM RL experiments.
Handles frame preprocessing and action mapping.
"""

import gymnasium as gym
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any

# Register ALE environments with gymnasium
import ale_py
gym.register_envs(ale_py)


class PongEnvironment:
    """Wrapper for Atari Pong environment with vision model-friendly preprocessing."""

    # Pong action space mapping (all 6 available actions)
    ACTIONS = {
        "NOOP": 0,          # Do nothing
        "FIRE": 1,          # Fire (start game)
        "RIGHT": 2,         # Move paddle right/up
        "LEFT": 3,          # Move paddle left/down
        "RIGHTFIRE": 4,     # Move right and fire
        "LEFTFIRE": 5,      # Move left and fire
    }

    ACTION_NAMES = {v: k for k, v in ACTIONS.items()}

    def __init__(self, render_mode: str = None, frame_skip: int = 1):
        """
        Initialize Pong environment.

        Args:
            render_mode: 'human' for visualization, 'rgb_array' for frame capture, None for headless
            frame_skip: Number of frames to repeat each action (default: 1 = no skip)
                       Higher values reduce LLM calls but may miss fast-changing game states
                       Recommended: 4-8 for Pong (60 FPS game, so 4 = ~15 decisions/sec)
        """
        self.env = gym.make('ALE/Pong-v5', render_mode=render_mode)
        self.frame_skip = max(1, frame_skip)  # Ensure at least 1
        self.current_step = 0
        self.episode_reward = 0
        self.done = False

    def reset(self) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Returns:
            frame: PIL Image of initial frame
            info: Additional environment info
        """
        obs, info = self.env.reset()
        self.current_step = 0
        self.episode_reward = 0
        self.done = False

        frame = self._preprocess_frame(obs)
        return frame, info

    def step(self, action: str) -> Tuple[Image.Image, float, bool, bool, Dict[str, Any]]:
        """
        Execute action in environment, repeating it for frame_skip steps.

        Args:
            action: Action name ('NOOP', 'FIRE', 'RIGHT', 'LEFT', etc.)

        Returns:
            frame: PIL Image of resulting frame (after frame_skip repetitions)
            reward: Total reward accumulated across all skipped frames
            terminated: Whether episode ended (game over)
            truncated: Whether episode was truncated (time limit)
            info: Additional environment info from last step
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action '{action}'. Must be one of {list(self.ACTIONS.keys())}")

        action_id = self.ACTIONS[action]
        total_reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}

        # Execute the same action for frame_skip steps
        for _ in range(self.frame_skip):
            obs, reward, terminated, truncated, info = self.env.step(action_id)
            total_reward += float(reward)

            self.current_step += 1

            # Stop early if episode ends
            if terminated or truncated:
                break

        self.episode_reward += total_reward
        self.done = terminated or truncated

        # obs is guaranteed to be set after at least one loop iteration
        frame = self._preprocess_frame(obs)  # type: ignore

        return frame, total_reward, terminated, truncated, info

    def _preprocess_frame(self, obs: np.ndarray) -> Image.Image:
        """
        Preprocess observation for vision model.

        Args:
            obs: Raw observation from environment (210, 160, 3) RGB array

        Returns:
            PIL Image suitable for SmolVLM (384x384 as per model specs)
        """
        # Convert numpy array to PIL Image
        img = Image.fromarray(obs)

        # Resize to SmolVLM's expected size (384x384)
        img = img.resize((384, 384), Image.Resampling.BILINEAR)

        return img

    def close(self):
        """Close the environment."""
        self.env.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get current episode statistics."""
        return {
            'steps': self.current_step,
            'total_reward': self.episode_reward,
            'done': self.done
        }

    @staticmethod
    def get_action_space() -> list[str]:
        """Get list of valid action names."""
        return list(PongEnvironment.ACTIONS.keys())


if __name__ == "__main__":
    # Test the environment
    print("Testing Pong Environment...")
    env = PongEnvironment()

    frame, info = env.reset()
    print(f"Initial frame size: {frame.size}")
    print(f"Available actions: {env.get_action_space()}")

    # Test a few random actions
    for i in range(5):
        action = np.random.choice(env.get_action_space())
        frame, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward}, done={terminated or truncated}")

    stats = env.get_stats()
    print(f"\nEpisode stats: {stats}")

    env.close()
    print("Environment test complete!")
