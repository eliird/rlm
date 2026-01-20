"""
Fine-tuning pipeline for SmolVLM on Pong gameplay data.
Uses full fine-tuning to update model weights based on reflection data.
"""

import os
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
import json


class PongDataset(torch.utils.data.Dataset):
    """Custom dataset for Pong training examples."""

    def __init__(self, examples: List[Dict[str, Any]], processor):
        """
        Initialize dataset.

        Args:
            examples: List of training examples with 'image' and 'messages'
            processor: HuggingFace processor for the model
        """
        self.examples = examples
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Extract image and conversation
        image = example['image']
        messages = example['messages']

        # Apply chat template
        text = self.processor.apply_chat_template(messages, add_generation_prompt=False)

        # Process inputs
        inputs = self.processor(
            text=text,
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        return inputs


class SmolVLMFinetuner:
    """Fine-tuner for SmolVLM on Pong gameplay."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
        output_dir: str = "data/checkpoints",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize fine-tuner.

        Args:
            model_name: Base model to fine-tune
            output_dir: Directory to save checkpoints
            device: Device for training
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        print(f"Initializing fine-tuner for: {model_name}")
        print(f"Output directory: {output_dir}")
        print(f"Device: {device}")

    def prepare_dataset(
        self,
        training_examples: List[Dict[str, Any]],
    ) -> PongDataset:
        """
        Prepare training dataset from examples.

        Args:
            training_examples: List of examples from reflection generator

        Returns:
            PongDataset ready for training
        """
        print(f"\nPreparing dataset with {len(training_examples)} examples...")

        # Load processor
        processor = AutoProcessor.from_pretrained(self.model_name)

        # Create dataset
        dataset = PongDataset(training_examples, processor)

        print(f"✓ Dataset prepared with {len(dataset)} examples")
        return dataset

    def finetune(
        self,
        training_examples: List[Dict[str, Any]],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5,
        save_steps: int = 100,
        logging_steps: int = 10,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        run_name: Optional[str] = None,
    ) -> str:
        """
        Fine-tune SmolVLM on training examples.

        Args:
            training_examples: List of training examples
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            learning_rate: Learning rate
            save_steps: Save checkpoint every N steps
            logging_steps: Log every N steps
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Warmup steps
            max_grad_norm: Max gradient norm for clipping
            run_name: Name for this training run

        Returns:
            Path to final checkpoint
        """
        if run_name is None:
            run_name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"\n{'='*70}")
        print(f"Starting fine-tuning: {run_name}")
        print(f"{'='*70}")
        print(f"Training examples: {len(training_examples)}")
        print(f"Epochs: {num_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"{'='*70}\n")

        # Load model and processor
        print("Loading model and processor...")
        processor = AutoProcessor.from_pretrained(self.model_name)
        model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )

        # Enable training mode
        model.train()

        print("✓ Model loaded")

        # Prepare dataset
        train_dataset = self.prepare_dataset(training_examples)

        # Training arguments
        output_path = self.output_dir / run_name
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            max_grad_norm=max_grad_norm,
            logging_steps=logging_steps,
            save_steps=save_steps,
            save_total_limit=3,
            bf16=True,  # Use bfloat16 for A100
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            report_to="tensorboard",
            logging_dir=str(output_path / "logs"),
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=processor.tokenizer,
        )

        # Train
        print("\nStarting training...")
        trainer.train()

        # Save final model
        final_path = output_path / "final"
        trainer.save_model(str(final_path))
        processor.save_pretrained(str(final_path))

        print(f"\n{'='*70}")
        print(f"✓ Fine-tuning complete!")
        print(f"Model saved to: {final_path}")
        print(f"{'='*70}\n")

        return str(final_path)

    def evaluate_checkpoint(
        self,
        checkpoint_path: str,
        test_examples: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate a checkpoint on test examples.

        Args:
            checkpoint_path: Path to model checkpoint
            test_examples: List of test examples

        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating checkpoint: {checkpoint_path}")

        # Load model
        processor = AutoProcessor.from_pretrained(checkpoint_path)
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
        )
        model.eval()

        # Simple accuracy metric
        correct = 0
        total = 0

        with torch.no_grad():
            for example in test_examples:
                messages = example['messages']
                image = example['image']

                # Get ground truth action
                gt_response = messages[1]['content']  # Assistant response
                gt_action = json.loads(gt_response)['action']

                # Generate prediction
                text = processor.apply_chat_template(messages[:-1], add_generation_prompt=True)
                inputs = processor(text=text, images=[image], return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                generated_ids = model.generate(**inputs, max_new_tokens=100)
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # Parse predicted action
                try:
                    pred_data = json.loads(generated_text)
                    pred_action = pred_data.get('action', '').upper()
                except:
                    pred_action = 'UNKNOWN'

                if pred_action == gt_action:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
        }

        print(f"Evaluation results:")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Correct: {correct}/{total}")

        return metrics


if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("See main.py for usage examples.")
