# Main experiment runner
# Tests small model on all questions and records results

import pandas as pd
import requests
from src.config import SMALL_MODEL, OLLAMA_BASE_URL, N_SAMPLES
import os


def call_small_model(prompt: str) -> str:
    """Call local Ollama model."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": SMALL_MODEL,
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"].strip()


def run_experiment():
    """Run small model on English questions dataset."""
    df = pd.read_csv("dataset/questions.csv")
    results = []

    for _, row in df.iterrows():
        print(f"Question {row['id']}: {row['english_question'][:50]}...")
        answers = [call_small_model(row['english_question']) 
                   for _ in range(N_SAMPLES)]
        
        results.append({
            "id": row["id"],
            "question": row["english_question"],
            "correct_answer": row["correct_answer"],
            "model_answers": answers,
            "language": "english"
        })

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(results).to_csv("results/english_results.csv", index=False)
    print("Done! Results saved.")


if __name__ == "__main__":
    run_experiment()