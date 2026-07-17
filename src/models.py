# Model adapters — all LLM calls go through here
# Keeps the rest of the code model-agnostic

import requests
import time
from groq import Groq
from src.config import (
    SMALL_MODEL, LARGE_MODEL,
    OLLAMA_BASE_URL, GROQ_API_KEY
)

groq_client = Groq(api_key=GROQ_API_KEY)

def call_small_model(prompt: str) -> str:
    """Call local small model via Ollama."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": SMALL_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"].strip()


def call_large_model(prompt: str) -> str:
    """Call large model via Groq API."""
    # Small delay to respect rate limits
    time.sleep(1.5)
    response = groq_client.chat.completions.create(
        model=LARGE_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()