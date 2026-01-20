#!/usr/bin/env python3
"""
Main training loop for LLM-based Reinforcement Learning on Atari Pong.

This implements the trial-reflect-finetune loop:
1. Run N trial episodes with current model
2. Reflect on poor outcomes to generate corrected training data
3. Fine-tune model on corrected data
4. Repeat and measure improvement

Usage:
    .venv/bin/python main.py --iterations 5 --episodes-per-iteration 100
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from src.environment.atari_wrapper import PongEnvironment
from src.models.smolvlm_agent import SmolVLMAgent
from src.training.trial_runner import TrialRunner
from src.training.reflection import ReflectionGenerator
from src.training.finetuner import SmolVLMFinetuner


class RLTrainingLoop:
    """Orchestrates the trial-reflect-finetune loop."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        output_dir: str = "data/experiments",
        experiment_name: str = None,
    ):
        """
        Initialize training loop.

        Args:
            model_name: Initial model to start from
            output_dir: Directory for experiment outputs
            experiment_name: Name for this experiment
        """
        if experiment_name is None:
            experiment_name = f"pong_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.current_model = model_name
        self.iteration_history = []

        print(f"\n{'='*80}")
        print(f"LLM-RL Training Loop Initialized")
        print(f"{'='*80}")
        print(f"Experiment: {experiment_name}")
        print(f"Output dir: {self.output_dir}")
        print(f"Base model: {model_name}")
        print(f"{'='*80}\n")

    def run_iteration(
        self,
        iteration: int,
        num_episodes: int = 100,
        max_steps_per_episode: int = 1000,
        reflection_sample_rate: float = 0.3,
        positive_sample_rate: float = 1.0,
        finetune_epochs: int = 3,
        finetune_batch_size: int = 4,
        finetune_lr: float = 2e-5,
        verbose: bool = False,
        frame_skip: int = 4,
    ):
        """
        Run one iteration of trial-reflect-finetune.

        Args:
            iteration: Iteration number
            num_episodes: Number of trial episodes
            max_steps_per_episode: Max steps per episode
            reflection_sample_rate: Fraction of poor steps to reflect on
            positive_sample_rate: Fraction of successful steps to reflect on (for spatial learning)
            finetune_epochs: Epochs for fine-tuning
            finetune_batch_size: Batch size for fine-tuning
            finetune_lr: Learning rate for fine-tuning
            verbose: Enable verbose logging
            frame_skip: Number of frames to repeat each action (reduces LLM calls)

        Returns:
            Dictionary with iteration results
        """
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration}")
        print(f"{'='*80}\n")

        iteration_dir = self.output_dir / f"iteration_{iteration}"
        iteration_dir.mkdir(exist_ok=True)

        # 1. TRIAL PHASE: Run episodes with current model
        print(f"PHASE 1: Running {num_episodes} trial episodes...")
        print(f"Model: {self.current_model}")
        print(f"Frame skip: {frame_skip} (action repeated {frame_skip}x per decision)")
        if iteration == 1:
            print(f"Note: Early iteration - training frequently with fewer episodes")
        print(f"Verbose logging: {verbose}\n")

        agent = SmolVLMAgent(model_name=self.current_model)
        env = PongEnvironment(frame_skip=frame_skip)
        runner = TrialRunner(agent, env, save_dir=str(iteration_dir / "episodes"))

        episodes = runner.run_trials(
            num_episodes=num_episodes,
            max_steps_per_episode=max_steps_per_episode,
            trial_name=f"iteration_{iteration}",
            verbose=verbose,
        )

        trial_stats = runner.get_trial_statistics(episodes)

        env.close()
        del agent  # Free GPU memory

        # 2. REFLECTION PHASE: Generate corrected training data
        print(f"\nPHASE 2: Generating reflections...")

        agent = SmolVLMAgent(model_name=self.current_model)
        reflector = ReflectionGenerator(agent, output_dir=str(iteration_dir / "training_data"))

        training_examples = reflector.generate_reflections(
            episodes=episodes,
            reward_threshold=0.0,  # Threshold for negative examples
            sample_rate=reflection_sample_rate,  # Sample rate for negative/neutral
            positive_sample_rate=positive_sample_rate,  # Sample rate for positive (scores)
        )

        # Save reflection statistics
        reflection_stats = reflector.analyze_training_data(training_examples)
        reflector.save_training_data(training_examples, filename=f"iteration_{iteration}.json")

        # Create fine-tuning dataset
        finetuning_data = reflector.create_finetuning_dataset(training_examples)

        del agent  # Free GPU memory

        # 3. FINE-TUNING PHASE: Update model weights
        print(f"\nPHASE 3: Fine-tuning model...")

        if len(finetuning_data) < 10:
            print(f"Warning: Only {len(finetuning_data)} training examples. Skipping fine-tuning.")
            checkpoint_path = self.current_model
        else:
            finetuner = SmolVLMFinetuner(
                model_name=self.current_model,
                output_dir=str(iteration_dir / "checkpoints"),
            )

            checkpoint_path = finetuner.finetune(
                training_examples=finetuning_data,
                num_epochs=finetune_epochs,
                batch_size=finetune_batch_size,
                learning_rate=finetune_lr,
                run_name=f"iteration_{iteration}",
            )

            # Update current model for next iteration
            self.current_model = checkpoint_path

        # 4. Save iteration results
        iteration_results = {
            'iteration': iteration,
            'trial_statistics': trial_stats,
            'reflection_statistics': reflection_stats,
            'num_training_examples': len(training_examples),
            'checkpoint_path': checkpoint_path,
        }

        results_path = iteration_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(iteration_results, f, indent=2)

        self.iteration_history.append(iteration_results)

        print(f"\n{'='*80}")
        print(f"ITERATION {iteration} COMPLETE")
        print(f"{'='*80}")
        print(f"Trial episodes: {num_episodes}")
        print(f"Mean reward: {trial_stats['mean_reward']:.2f} ± {trial_stats['std_reward']:.2f}")
        print(f"Training examples: {len(training_examples)}")
        print(f"New checkpoint: {checkpoint_path}")
        print(f"{'='*80}\n")

        return iteration_results

    def run_training_loop(
        self,
        num_iterations: int = 5,
        episodes_per_iteration: int = 100,
        adaptive_episodes: bool = False,
        initial_episodes: int = 2,
        **kwargs
    ):
        """
        Run the complete training loop for multiple iterations.

        Args:
            num_iterations: Number of trial-reflect-finetune iterations
            episodes_per_iteration: Episodes per iteration (used if adaptive_episodes=False)
            adaptive_episodes: If True, start with fewer episodes and increase over time
            initial_episodes: Starting number of episodes (when adaptive_episodes=True)
            **kwargs: Additional arguments passed to run_iteration
        """
        print(f"\n{'='*80}")
        print(f"Starting Training Loop")
        print(f"{'='*80}")
        print(f"Iterations: {num_iterations}")
        if adaptive_episodes:
            print(f"Adaptive episodes: Starting with {initial_episodes}, doubling each iteration")
        else:
            print(f"Episodes per iteration: {episodes_per_iteration}")
        print(f"{'='*80}\n")

        for iteration in range(1, num_iterations + 1):
            # Calculate episodes for this iteration
            if adaptive_episodes:
                # Start small, double each iteration: 2, 4, 8, 16, ...
                # Cap at episodes_per_iteration
                num_episodes = min(initial_episodes * (2 ** (iteration - 1)), episodes_per_iteration)
            else:
                num_episodes = episodes_per_iteration

            self.run_iteration(
                iteration=iteration,
                num_episodes=num_episodes,
                **kwargs
            )

        # Save final summary
        self.save_experiment_summary()

        print(f"\n{'='*80}")
        print(f"TRAINING LOOP COMPLETE")
        print(f"{'='*80}")
        print(f"Total iterations: {num_iterations}")
        print(f"Experiment: {self.experiment_name}")
        print(f"Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")

    def save_experiment_summary(self):
        """Save overall experiment summary."""
        summary = {
            'experiment_name': self.experiment_name,
            'num_iterations': len(self.iteration_history),
            'final_model': self.current_model,
            'iteration_history': self.iteration_history,
        }

        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Experiment summary saved to: {summary_path}")

        # Print improvement summary
        if len(self.iteration_history) > 1:
            print("\nReward Progression:")
            for i, results in enumerate(self.iteration_history, 1):
                mean_reward = results['trial_statistics']['mean_reward']
                print(f"  Iteration {i}: {mean_reward:.2f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="LLM-RL Training on Atari Pong")
    parser.add_argument("--iterations", type=int, default=3, help="Number of training iterations")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per iteration")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    parser.add_argument("--reflection-rate", type=float, default=0.3, help="Reflection sample rate")
    parser.add_argument("--finetune-epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--finetune-batch-size", type=int, default=4, help="Fine-tuning batch size")
    parser.add_argument("--finetune-lr", type=float, default=2e-5, help="Fine-tuning learning rate")
    parser.add_argument("--experiment-name", type=str, default=None, help="Experiment name")

    args = parser.parse_args()

    # Create training loop
    loop = RLTrainingLoop(experiment_name=args.experiment_name)

    # Run training
    loop.run_training_loop(
        num_iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        max_steps_per_episode=args.max_steps,
        reflection_sample_rate=args.reflection_rate,
        finetune_epochs=args.finetune_epochs,
        finetune_batch_size=args.finetune_batch_size,
        finetune_lr=args.finetune_lr,
    )


if __name__ == "__main__":
    main()
