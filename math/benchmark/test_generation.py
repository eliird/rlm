"""
Quick generation test to verify the vLLM server is running and responding correctly.

Requires vLLM server running:
  bash math/benchmark/serve.sh

Run from repo root:
  python math/benchmark/test_generation.py
"""

import requests

SERVER_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "deepseek-r1-32b"

PROBLEMS = [
    "What is 2 + 2?",
    "Find all real solutions to x^2 - 5x + 6 = 0.",
    "What is the derivative of x^3 + 2x?",
]


def query(problem: str) -> tuple[str, str]:
    resp = requests.post(
        SERVER_URL,
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": problem}],
            "max_tokens": 1024,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = (resp.json()["choices"][0]["message"].get("content") or "").strip()
    if "</think>" in content:
        thinking, response = content.split("</think>", 1)
        thinking = thinking.replace("<think>", "").strip()
        response = response.strip()
    else:
        thinking, response = "", content
    return thinking, response


# Verify server is up
try:
    requests.get("http://localhost:8000/health", timeout=5).raise_for_status()
except Exception:
    print("ERROR: vLLM server not reachable. Run: bash math/benchmark/serve.sh")
    exit(1)

print(f"Server OK. Model: {MODEL_NAME}\n")

for problem in PROBLEMS:
    print("=" * 60)
    print(f"PROBLEM: {problem}")
    thinking, response = query(problem)
    print(f"THINKING: {thinking[:300]}{'...' if len(thinking) > 300 else ''}")
    print(f"RESPONSE:\n{response}\n")
