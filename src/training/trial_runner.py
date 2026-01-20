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

    def __init__(self, episode_id: int, frames_dir: Path = None):
        self.episode_id = episode_id
        self.steps: List[Dict[str, Any]] = []
        self.total_reward = 0.0
        self.duration = 0
        self.frames_dir = frames_dir

        # Create frames directory if specified
        if self.frames_dir:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

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
        """Add a step to the episode, saving frames to disk."""
        # Save frames to disk if frames_dir is set
        if self.frames_dir:
            frame_path = self.frames_dir / f"ep{self.episode_id}_step{step_num}_frame.png"
            next_frame_path = self.frames_dir / f"ep{self.episode_id}_step{step_num}_next.png"

            frame.save(frame_path)
            next_frame.save(next_frame_path)

            # Store only the file paths, not the images
            frame_ref = str(frame_path)
            next_frame_ref = str(next_frame_path)
        else:
            # Keep images in memory (legacy behavior)
            frame_ref = frame
            next_frame_ref = next_frame

        self.steps.append({
            'step': step_num,
            'frame': frame_ref,
            'action': action,
            'reasoning': reasoning,
            'reward': reward,
            'next_frame': next_frame_ref,
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
        save_frames_to_disk: bool = True,
    ):
        """
        Initialize trial runner.

        Args:
            agent: SmolVLM agent
            env: Pong environment
            save_dir: Directory to save episode data
            save_frames_to_disk: If True, save frames to disk instead of keeping in RAM
        """
        self.agent = agent
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_frames_to_disk = save_frames_to_disk

        # Create frames directory if saving to disk
        if self.save_frames_to_disk:
            self.frames_dir = self.save_dir / "frames"
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.frames_dir = None

        self.episodes: List[Episode] = []

    def run_episode(
        self,
        episode_id: int,
        max_steps: int = 1000,
        temperature: float = 0.7,
        render: bool = False,
        verbose: bool = False,
    ) -> Episode:
        """
        Run a single episode.

        Args:
            episode_id: Unique episode identifier
            max_steps: Maximum steps per episode
            temperature: Sampling temperature for agent
            render: Whether to render the environment
            verbose: Print detailed step information

        Returns:
            Episode object with all collected data
        """
        episode = Episode(episode_id, frames_dir=self.frames_dir)

        # Reset environment
        frame, info = self.env.reset()

        if verbose:
            print(f"\n  Starting Episode {episode_id}")

        for step in range(max_steps):
            # Get action from agent (LLM inference - this is the slow part)
            if verbose and step % 10 == 0:
                print(f"    Step {step}: Getting action from LLM...")

            action, reasoning, raw_output = self.agent.get_action(
                frame,
                game_context=f"Step {step}",
                temperature=temperature,
            )

            if verbose and step % 10 == 0:
                print(f"    Step {step}: Action={action}, Reasoning={reasoning[:50]}...")

            # Execute action
            next_frame, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            if verbose and reward != 0:
                print(f"    Step {step}: REWARD={reward}!")

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
                if verbose:
                    print(f"  Episode {episode_id} finished at step {step}. Total reward: {episode.total_reward}")
                break

        if verbose and not done:
            print(f"  Episode {episode_id} reached max steps ({max_steps}). Total reward: {episode.total_reward}")

        return episode

    def run_trials(
        self,
        num_episodes: int,
        max_steps_per_episode: int = 1000,
        temperature: float = 0.7,
        save_frequency: int = 10,
        trial_name: str = None,
        verbose: bool = False,
    ) -> List[Episode]:
        """
        Run multiple trial episodes.

        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Max steps per episode
            temperature: Sampling temperature
            save_frequency: Save data every N episodes
            trial_name: Name for this trial batch (defaults to timestamp)
            verbose: Enable verbose logging for each episode

        Returns:
            List of Episode objects
        """
        if trial_name is None:
            trial_name = datetime.now().strftime("%Y%m%d_%H%M%S")

        print(f"\n{'='*60}")
        print(f"Running {num_episodes} trial episodes")
        print(f"Trial name: {trial_name}")
        print(f"Max steps per episode: {max_steps_per_episode}")
        print(f"{'='*60}\n")

        episodes = []
        stats = {
            'rewards': [],
            'durations': [],
        }

        # Use tqdm only if not verbose (to avoid clutter)
        episode_iterator = range(num_episodes) if verbose else tqdm(range(num_episodes), desc="Episodes")

        for ep_num in episode_iterator:
            if verbose:
                print(f"\n{'─'*60}")
                print(f"Episode {ep_num + 1}/{num_episodes}")
                print(f"{'─'*60}")

            episode = self.run_episode(
                episode_id=ep_num,
                max_steps=max_steps_per_episode,
                temperature=temperature,
                verbose=verbose,
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
