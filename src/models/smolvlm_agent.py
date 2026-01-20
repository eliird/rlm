"""
SmolVLM agent for Atari game playing with structured JSON output.
Handles vision input and outputs reasoning + action decisions.
"""

import os
import sys
from pathlib import Path

# Set HuggingFace cache to local directory BEFORE importing transformers
# This must happen before any transformers import
_project_root = Path(__file__).parent.parent.parent
_cache_dir = str(_project_root / 'data' / 'hf_cache')
os.environ['HF_HOME'] = _cache_dir
os.environ['TRANSFORMERS_CACHE'] = _cache_dir
os.environ['HF_HUB_CACHE'] = _cache_dir

import json
import re
from typing import Dict, Optional, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq


class SmolVLMAgent:
    """SmolVLM-based agent that outputs structured action decisions."""

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize SmolVLM agent.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Model data type (bfloat16 recommended for A100)
        """
        self.device = device
        self.dtype = dtype
        self.model_name = model_name

        print(f"Loading SmolVLM model: {model_name}")
        print(f"Device: {device}, dtype: {dtype}")

        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()

        print("✓ Model loaded successfully")
        print("\nNote: The base model may not play well initially.")
        print("It will learn through the trial-reflect-finetune loop.")

        # System prompt for Pong game with concrete examples
        self.system_prompt = """You are playing Pong. Your paddle is on the RIGHT side of the screen.

ACTIONS: NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE

Analyze the image and respond with JSON only.

Examples:
- If ball is in upper area: {"reasoning": "Ball is high, moving paddle up", "action": "RIGHT"}
- If ball is in lower area: {"reasoning": "Ball is low, moving paddle down", "action": "LEFT"}
- If game not started: {"reasoning": "Starting game", "action": "FIRE"}

Your turn - analyze the image and respond with JSON:"""

    def get_action(
        self,
        frame: Image.Image,
        game_context: Optional[str] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 100,
    ) -> Tuple[str, str, str]:
        """
        Get action decision from the model given a game frame.

        Args:
            frame: PIL Image of current game state
            game_context: Optional context (e.g., "Game just started", "Current score: 5-3")
            temperature: Sampling temperature for generation
            max_new_tokens: Maximum tokens to generate

        Returns:
            Tuple of (action, reasoning, raw_output)
        """
        # Build prompt - be very direct
        user_message = f"{self.system_prompt}\n\nAnalyze this Pong game frame and choose your action. Respond with JSON only."

        # Create messages in chat format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_message}
                ]
            }
        ]

        # Process inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[frame], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        # Decode only the newly generated tokens (not the input prompt)
        input_length = inputs['input_ids'].shape[1]
        new_tokens = generated_ids[:, input_length:]

        generated_texts = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        raw_output = generated_texts[0]

        # Parse JSON from output
        action, reasoning = self._parse_output(raw_output)

        return action, reasoning, raw_output

    def _parse_output(self, output: str) -> Tuple[str, str]:
        """
        Parse JSON output to extract action and reasoning.

        Args:
            output: Raw model output string

        Returns:
            Tuple of (action, reasoning)
        """
        # Try to find JSON in the output
        json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                action = data.get("action", "NOOP").upper()
                reasoning = data.get("reasoning", "No reasoning provided")

                # Validate action
                valid_actions = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
                if action not in valid_actions:
                    print(f"Warning: Invalid action '{action}', defaulting to NOOP")
                    action = "NOOP"

                return action, reasoning

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse JSON: {e}")
                print(f"Output was: {output}")

        # Fallback: try to extract action from text
        print(f"Warning: No valid JSON found in output, using fallback parsing")
        print(f"Output was: {output}")

        for action in ["RIGHTFIRE", "LEFTFIRE", "RIGHT", "LEFT", "FIRE", "NOOP"]:
            if action.lower() in output.lower():
                return action, f"Extracted from text: {output[:100]}"

        return "NOOP", f"Could not parse output: {output[:100]}"

    def create_reflection_prompt(
        self,
        frame: Image.Image,
        action_taken: str,
        reasoning: str,
        reward: float,
        next_frame: Image.Image,
    ) -> str:
        """
        Create a prompt for the model to reflect on its action and generate better training data.

        Args:
            frame: Original frame where action was taken
            action_taken: Action that was executed
            reasoning: Original reasoning for the action
            reward: Reward received after action
            next_frame: Resulting frame after action

        Returns:
            Prompt for self-reflection
        """
        # Create reward-specific guidance
        if reward < 0:
            guidance = "You LOST a point! The ball got past your paddle. What went wrong?"
        elif reward > 0:
            guidance = "You SCORED a point! This was good. What made it work?"
        else:
            guidance = "No score change. Track the ball position - where is it and where should your paddle be?"

        valid_actions = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

        prompt = f"""Analyze this Pong game. {guidance}

Previous action: {action_taken}
Previous reasoning: {reasoning}
Result: Reward {reward}

Your paddle is on the RIGHT side of the screen.
- RIGHT = move paddle UP
- LEFT = move paddle DOWN
- FIRE = start game
- NOOP = do nothing

Look at both images. Where is the ball? Where is your paddle? What should you do?

Respond with JSON only:
{{
  "analysis": "Describe the ball and paddle positions you see",
  "correct_action": "Choose: RIGHT, LEFT, FIRE, NOOP, RIGHTFIRE, or LEFTFIRE",
  "correct_reasoning": "Explain why this move is best"
}}

Your correct_action MUST be one word: RIGHT, LEFT, FIRE, NOOP, RIGHTFIRE, or LEFTFIRE"""

        return prompt

    def get_reflection(
        self,
        frame: Image.Image,
        action_taken: str,
        reasoning: str,
        reward: float,
        next_frame: Image.Image,
        temperature: float = 0.7,
        max_new_tokens: int = 300,
    ) -> Dict[str, str]:
        """
        Get model's reflection on a previous action to generate better training data.

        Args:
            frame: Original frame
            action_taken: Action that was executed
            reasoning: Original reasoning
            reward: Reward received
            next_frame: Frame after action
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with 'analysis', 'correct_action', 'correct_reasoning'
        """
        reflection_prompt = self.create_reflection_prompt(
            frame, action_taken, reasoning, reward, next_frame
        )

        # Create messages with both frames
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ORIGINAL FRAME (before your action):"},
                    {"type": "image"},  # frame
                    {"type": "text", "text": "RESULT FRAME (after your action):"},
                    {"type": "image"},  # next_frame
                    {"type": "text", "text": reflection_prompt}
                ]
            }
        ]

        # Process inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[frame, next_frame], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )

        # Decode only the newly generated tokens (not the input prompt)
        input_length = inputs['input_ids'].shape[1]
        new_tokens = generated_ids[:, input_length:]

        generated_texts = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        raw_output = generated_texts[0]

        # Parse reflection JSON
        return self._parse_reflection(raw_output)

    def _parse_reflection(self, output: str) -> Dict[str, str]:
        """Parse reflection output."""
        import random
        valid_actions = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

        json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                correct_action = data.get('correct_action', 'NOOP').upper().strip()

                # Validate and fix invalid actions
                if correct_action not in valid_actions:
                    # Check if it contains a valid action
                    found = False
                    for action in valid_actions:
                        if action in correct_action:
                            correct_action = action
                            found = True
                            break

                    # If still invalid, use a random action (not NOOP to encourage exploration)
                    if not found:
                        original_action = data.get('correct_action', 'NONE')
                        correct_action = random.choice(["FIRE", "RIGHT", "LEFT"])
                        print(f"  Warning: Invalid reflection action '{original_action}', using random: {correct_action}")
                        if len(output) < 500:
                            print(f"  Raw output: {output}")

                return {
                    'analysis': data.get('analysis', 'No analysis'),
                    'correct_action': correct_action,
                    'correct_reasoning': data.get('correct_reasoning', 'No reasoning')
                }
            except json.JSONDecodeError:
                pass

        # Fallback - use random action to encourage diversity
        return {
            'analysis': output[:200],
            'correct_action': random.choice(["FIRE", "RIGHT", "LEFT"]),
            'correct_reasoning': 'Failed to parse reflection - using random action'
        }


if __name__ == "__main__":
    # Test the agent
    print("Testing SmolVLM Agent...")

    # Create a dummy image
    import numpy as np
    dummy_frame = Image.fromarray(np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8))

    agent = SmolVLMAgent()

    print("\nTesting action generation...")
    action, reasoning, raw = agent.get_action(dummy_frame, "Game just started")
    print(f"Action: {action}")
    print(f"Reasoning: {reasoning}")
    print(f"Raw output: {raw[:200]}...")

    print("\n✓ Agent test complete!")
