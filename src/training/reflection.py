"""
Self-reflection mechanism for generating improved training data.
Analyzes episodes and creates corrected training examples.
"""

import json
from typing import List, Dict, Any, Union
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from ..models.smolvlm_agent import SmolVLMAgent


def load_frame(frame_ref: Union[str, Image.Image]) -> Image.Image:
    """
    Load a frame from disk if it's a path, otherwise return as-is.

    Args:
        frame_ref: Either a file path (str) or PIL Image

    Returns:
        PIL Image
    """
    if isinstance(frame_ref, str):
        return Image.open(frame_ref)
    return frame_ref


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
        positive_sample_rate: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Generate reflections for both successful and unsuccessful actions.

        Args:
            episodes: List of Episode objects from trial runner
            reward_threshold: Threshold for negative examples (reflect on reward <= this)
            sample_rate: Fraction of negative/neutral steps to reflect on (0.0 to 1.0)
            positive_sample_rate: Fraction of positive steps to reflect on (0.0 to 1.0)
                                 Use 1.0 initially for spatial learning, can reduce later

        Returns:
            List of training examples with corrected actions
        """
        print(f"\nGenerating reflections from {len(episodes)} episodes...")
        print(f"  Negative/neutral sample rate: {sample_rate:.1%}")
        print(f"  Positive sample rate: {positive_sample_rate:.1%}")

        training_examples = []

        for episode in tqdm(episodes, desc="Processing episodes"):
            for step_data in episode.steps:
                # Skip if done (can't get next frame for reflection)
                if step_data['done']:
                    continue

                reward = step_data['reward']

                # Determine if we should reflect on this step
                import random
                should_reflect = False

                if reward > reward_threshold:
                    # Positive reward - helps model learn spatial awareness
                    if random.random() <= positive_sample_rate:
                        should_reflect = True
                else:
                    # Negative or neutral - learn from mistakes
                    if random.random() <= sample_rate:
                        should_reflect = True

                if not should_reflect:
                    continue

                # Generate reflection
                try:
                    # Load frames from disk if they are stored as paths
                    frame = load_frame(step_data['frame'])
                    next_frame = load_frame(step_data['next_frame'])

                    reflection = self.agent.get_reflection(
                        frame=frame,
                        action_taken=step_data['action'],
                        reasoning=step_data['reasoning'],
                        reward=reward,
                        next_frame=next_frame,
                    )

                    # Create training example
                    # Keep frame reference (path or image) for later use
                    training_example = {
                        'frame': step_data['frame'],  # This will be a path if saving to disk
                        'original_action': step_data['action'],
                        'original_reasoning': step_data['reasoning'],
                        'reward': reward,
                        'reflection_analysis': reflection['analysis'],
                        'corrected_action': reflection['correct_action'],
                        'corrected_reasoning': reflection['correct_reasoning'],
                        'episode_id': episode.episode_id,
                        'step': step_data['step'],
                        'is_positive': reward > 0,  # Track if this was a successful action
                    }

                    training_examples.append(training_example)

                except Exception as e:
                    print(f"Warning: Reflection failed for episode {episode.episode_id}, step {step_data['step']}: {e}")
                    continue

        # Summary statistics
        num_positive = sum(1 for ex in training_examples if ex.get('is_positive', False))
        num_negative = len(training_examples) - num_positive

        print(f"✓ Generated {len(training_examples)} reflection-based training examples")
        print(f"  Positive examples (reward > 0): {num_positive}")
        print(f"  Negative/neutral examples: {num_negative}")

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

        # Convert to JSON-serializable format
        serializable_data = []
        for example in training_examples:
            # Handle frame reference (might be a path string or PIL image)
            frame_info = example['frame'] if isinstance(example['frame'], str) else 'in_memory'

            serializable_data.append({
                'frame_path': frame_info,
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
            # Load frame from disk if it's a path
            frame = load_frame(example['frame'])

            # Format as chat conversation for fine-tuning
            finetuning_example = {
                'image': frame,  # PIL Image (loaded from disk if needed)
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
