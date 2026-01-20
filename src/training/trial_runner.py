"""
Trial runner for collecting episodes of Pong gameplay.
Runs the agent through multiple episodes and records experiences.
"""

import os
import json
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

from ..environment.atari_wrapper import PongEnvironment
from ..models.smolvlm_agent import SmolVLMAgent


class Episode:
    """Stores data for a single episode."""

    def __init__(self, episode_id: int):
        self.episode_id = episode_id
        self.steps: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self.duration = 0

    def add_step(
        self,
        step_num: int,
        frame,  # PIL Image
        action: str,
        reasoning: str,
        reward: float,
        next_frame,  # PIL Image
        done: bool,
        raw_output: str = "",
    ):
        """Add a step to the episode."""
        self.steps.append({
            'step': step_num,
            'frame': frame,  # We'll save these separately
            'action': action,
            'reasoning': reasoning,
            'reward': reward,
            'next_frame': next_frame,
            'done': done,
            'raw_output': raw_output,
        })
        self.total_reward += reward
        self.duration += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary (without images)."""
        return {
            'episode_id': self.episode_id,
            'total_reward': self.total_reward,
            'duration': self.duration,
            'steps': [
                {
                    'step': s['step'],
                    'action': s['action'],
                    'reasoning': s['reasoning'],
                    'reward': s['reward'],
                    'done': s['done'],
                    'raw_output': s['raw_output'],
                }
                for s in self.steps
            ]
        }


class TrialRunner:
    """Runs trials and collects episode data."""

    def __init__(
        self,
        agent: SmolVLMAgent,
        env: PongEnvironment,
        save_dir: str = "data/episodes",
    ):
        """
        Initialize trial runner.

        Args:
            agent: SmolVLM agent
            env: Pong environment
            save_dir: Directory to save episode data
        """
        self.agent = agent
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.episodes: List[Episode] = []

    def run_episode(
        self,
        episode_id: int,
        max_steps: int = 1000,
        temperature: float = 0.7,
        render: bool = False,
    ) -> Episode:
        """
        Run a single episode.

        Args:
            episode_id: Unique episode identifier
            max_steps: Maximum steps per episode
            temperature: Sampling temperature for agent
            render: Whether to render the environment

        Returns:
            Episode object with all collected data
        """
        episode = Episode(episode_id)

        # Reset environment
        frame, info = self.env.reset()

        for step in range(max_steps):
            # Get action from agent
            action, reasoning, raw_output = self.agent.get_action(
                frame,
                game_context=f"Step {step}",
                temperature=temperature,
            )

            # Execute action
            next_frame, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Record step
            episode.add_step(
                step_num=step,
                frame=frame,
                action=action,
                reasoning=reasoning,
                reward=reward,
                next_frame=next_frame,
                done=done,
                raw_output=raw_output,
            )

            # Update frame
            frame = next_frame

            if done:
                break

        return episode

    def run_trials(
        self,
        num_episodes: int,
        max_steps_per_episode: int = 1000,
        temperature: float = 0.7,
        save_frequency: int = 10,
        trial_name: str = None,
    ) -> List[Episode]:
        """
        Run multiple trial episodes.

        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Max steps per episode
            temperature: Sampling temperature
            save_frequency: Save data every N episodes
            trial_name: Name for this trial batch (defaults to timestamp)

        Returns:
            List of Episode objects
        """
        if trial_name is None:
            trial_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*60}")
        print(f"Running {num_episodes} trial episodes")
        print(f"Trial name: {trial_name}")
        print(f"{'='*60}\n")

        episodes = []
        stats = {
            'rewards': [],
            'durations': [],
        }

        for ep_num in tqdm(range(num_episodes), desc="Episodes"):
            episode = self.run_episode(
                episode_id=ep_num,
                max_steps=max_steps_per_episode,
                temperature=temperature,
            )

            episodes.append(episode)
            stats['rewards'].append(episode.total_reward)
            stats['durations'].append(episode.duration)

            # Print progress
            if (ep_num + 1) % 10 == 0:
                avg_reward = np.mean(stats['rewards'][-10:])
                avg_duration = np.mean(stats['durations'][-10:])
                print(f"\nEpisode {ep_num + 1}/{num_episodes}")
                print(f"  Last 10 avg reward: {avg_reward:.2f}")
                print(f"  Last 10 avg duration: {avg_duration:.1f}")

            # Save periodically
            if (ep_num + 1) % save_frequency == 0:
                self.save_episodes(episodes, trial_name)

        # Final save
        self.save_episodes(episodes, trial_name)

        # Print summary statistics
        print(f"\n{'='*60}")
        print(f"Trial Complete: {trial_name}")
        print(f"{'='*60}")
        print(f"Total episodes: {len(episodes)}")
        print(f"Average reward: {np.mean(stats['rewards']):.2f} ± {np.std(stats['rewards']):.2f}")
        print(f"Average duration: {np.mean(stats['durations']):.1f} ± {np.std(stats['durations']):.1f}")
        print(f"Best reward: {max(stats['rewards']):.2f}")
        print(f"Worst reward: {min(stats['rewards']):.2f}")
        print(f"{'='*60}\n")

        self.episodes.extend(episodes)
        return episodes

    def save_episodes(self, episodes: List[Episode], trial_name: str):
        """
        Save episode data to disk.

        Args:
            episodes: List of episodes to save
            trial_name: Name for this trial batch
        """
        trial_dir = self.save_dir / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Save episode metadata (JSON without images)
        metadata = {
            'trial_name': trial_name,
            'num_episodes': len(episodes),
            'episodes': [ep.to_dict() for ep in episodes],
        }

        metadata_path = trial_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save frames separately (optional - can be memory intensive)
        # For now, we'll skip saving frames and rely on replay if needed

        print(f"✓ Saved {len(episodes)} episodes to {trial_dir}")

    def load_episodes(self, trial_name: str) -> List[Dict[str, Any]]:
        """
        Load episode metadata from disk.

        Args:
            trial_name: Name of trial batch to load

        Returns:
            List of episode dictionaries
        """
        trial_dir = self.save_dir / trial_name
        metadata_path = trial_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"No trial data found at {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        return metadata['episodes']

    def get_trial_statistics(self, episodes: List[Episode]) -> Dict[str, Any]:
        """Calculate statistics for a set of episodes."""
        rewards = [ep.total_reward for ep in episodes]
        durations = [ep.duration for ep in episodes]

        return {
            'num_episodes': len(episodes),
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'mean_duration': float(np.mean(durations)),
            'std_duration': float(np.std(durations)),
        }


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("See main.py for usage examples.")
