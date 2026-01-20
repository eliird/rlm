#!/usr/bin/env python3
"""
Test SmolVLM's vision understanding of Pong frames.
This will help us understand if the model can actually see and interpret the game.

Usage:
    .venv/bin/python test_vision_understanding.py
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


def test_vision_understanding():
    """Test if model can understand Pong game frames."""
    print("=" * 70)
    print("Testing SmolVLM's Vision Understanding of Pong")
    print("=" * 70)

    # Initialize environment and agent
    print("\n1. Initializing environment and agent...")
    env = PongEnvironment()
    agent = SmolVLMAgent()

    # Get a frame
    frame, _ = env.reset()
    print(f"\n2. Captured Pong frame: {frame.size}")

    # Save the frame for inspection
    frame_path = "data/test_frame.png"
    os.makedirs("data", exist_ok=True)
    frame.save(frame_path)
    print(f"   Saved frame to: {frame_path}")

    # Ask different questions about the frame
    questions = [
        "Describe what you see in this image in detail.",
        "What game is this? Describe the visual elements.",
        "Where is the ball located in this image?",
        "Where are the paddles located?",
        "What colors do you see in this game screen?",
        "Is this image from a video game? If so, which one?",
    ]

    print("\n3. Asking the model to analyze the frame...\n")
    print("=" * 70)

    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}: {question}")
        print("-" * 70)

        # Create a simple vision-text prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            }
        ]

        # Process inputs
        prompt = agent.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = agent.processor(text=prompt, images=[frame], return_tensors="pt")
        inputs = {k: v.to(agent.device) for k, v in inputs.items()}

        # Generate response
        import torch
        with torch.no_grad():
            generated_ids = agent.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.3,  # Lower temperature for more factual responses
                do_sample=True,
            )

        # Decode output
        generated_texts = agent.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        response = generated_texts[0]
        print(f"Response: {response}")
        print()

    env.close()

    print("=" * 70)
    print("Test complete!")
    print(f"Frame saved at: {frame_path}")
    print("=" * 70)


if __name__ == "__main__":
    test_vision_understanding()
