"""
Self-reflection mechanism for generating improved training data.
Analyzes episodes and creates corrected training examples.
"""

import json
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

from ..models.smolvlm_agent import SmolVLMAgent


class ReflectionGenerator:
    """Generates reflection-based training data from episode experiences."""

    def __init__(self, agent: SmolVLMAgent, output_dir: str = "data/training_data"):
        """
        Initialize reflection generator.

        Args:
            agent: SmolVLM agent for generating reflections
            output_dir: Directory to save training data
        """
        self.agent = agent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_reflections(
        self,
        episodes: List[Any],
        reward_threshold: float = 0.0,
        sample_rate: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate reflections for episodes where agent could improve.

        Args:
            episodes: List of Episode objects from trial runner
            reward_threshold: Only reflect on steps with reward <= this threshold
            sample_rate: Fraction of eligible steps to reflect on (0.0 to 1.0)

        Returns:
            List of training examples with corrected actions
        """
        print(f"\nGenerating reflections from {len(episodes)} episodes...")

        training_examples = []

        for episode in tqdm(episodes, desc="Processing episodes"):
            for step_data in episode.steps:
                # Skip if done (can't get next frame for reflection)
                if step_data['done']:
                    continue

                reward = step_data['reward']

                # Only reflect on poor outcomes or neutral ones
                if reward <= reward_threshold:
                    # Sample based on rate
                    import random
                    if random.random() > sample_rate:
                        continue

                    # Generate reflection
                    try:
                        reflection = self.agent.get_reflection(
                            frame=step_data['frame'],
                            action_taken=step_data['action'],
                            reasoning=step_data['reasoning'],
                            reward=reward,
                            next_frame=step_data['next_frame'],
                        )

                        # Create training example
                        training_example = {
                            'frame': step_data['frame'],
                            'original_action': step_data['action'],
                            'original_reasoning': step_data['reasoning'],
                            'reward': reward,
                            'reflection_analysis': reflection['analysis'],
                            'corrected_action': reflection['correct_action'],
                            'corrected_reasoning': reflection['correct_reasoning'],
                            'episode_id': episode.episode_id,
                            'step': step_data['step'],
                        }

                        training_examples.append(training_example)

                    except Exception as e:
                        print(f"Warning: Reflection failed for episode {episode.episode_id}, step {step_data['step']}: {e}")
                        continue

        print(f"✓ Generated {len(training_examples)} reflection-based training examples")
        return training_examples

    def save_training_data(
        self,
        training_examples: List[Dict[str, Any]],
        filename: str = "training_data.json",
    ):
        """
        Save training data to disk (without images).

        Args:
            training_examples: List of training examples
            filename: Output filename
        """
        output_path = self.output_dir / filename

        # Convert to JSON-serializable format (exclude PIL images)
        serializable_data = []
        for example in training_examples:
            serializable_data.append({
                'original_action': example['original_action'],
                'original_reasoning': example['original_reasoning'],
                'reward': example['reward'],
                'reflection_analysis': example['reflection_analysis'],
                'corrected_action': example['corrected_action'],
                'corrected_reasoning': example['corrected_reasoning'],
                'episode_id': example['episode_id'],
                'step': example['step'],
            })

        with open(output_path, 'w') as f:
            json.dump({
                'num_examples': len(serializable_data),
                'examples': serializable_data
            }, f, indent=2)

        print(f"✓ Saved training data to {output_path}")

    def create_finetuning_dataset(
        self,
        training_examples: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert training examples to fine-tuning format.

        Each example is formatted as an image-text pair for fine-tuning:
        - Input: game frame + system prompt
        - Output: corrected action and reasoning

        Args:
            training_examples: List of reflection-based examples

        Returns:
            List of formatted training examples ready for fine-tuning
        """
        finetuning_data = []

        for example in training_examples:
            # Format as chat conversation for fine-tuning
            finetuning_example = {
                'image': example['frame'],  # PIL Image
                'messages': [
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image'},
                            {'type': 'text', 'text': self.agent.system_prompt + '\n\nAnalyze this Pong game frame and choose your action. Respond with JSON only.'}
                        ]
                    },
                    {
                        'role': 'assistant',
                        'content': json.dumps({
                            'reasoning': example['corrected_reasoning'],
                            'action': example['corrected_action']
                        })
                    }
                ],
                'metadata': {
                    'episode_id': example['episode_id'],
                    'step': example['step'],
                    'original_action': example['original_action'],
                    'reward': example['reward'],
                }
            }

            finetuning_data.append(finetuning_example)

        return finetuning_data

    def analyze_training_data(
        self,
        training_examples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Analyze training data statistics.

        Args:
            training_examples: List of training examples

        Returns:
            Dictionary with statistics
        """
        if not training_examples:
            return {'num_examples': 0}

        # Count action corrections
        action_changes = {}
        for example in training_examples:
            original = example['original_action']
            corrected = example['corrected_action']
            key = f"{original} -> {corrected}"
            action_changes[key] = action_changes.get(key, 0) + 1

        # Reward distribution
        rewards = [ex['reward'] for ex in training_examples]
        import numpy as np

        stats = {
            'num_examples': len(training_examples),
            'action_changes': action_changes,
            'reward_distribution': {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
            },
            'unique_episodes': len(set(ex['episode_id'] for ex in training_examples)),
        }

        return stats


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("See main.py for usage examples.")
