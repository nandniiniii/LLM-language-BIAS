from datasets import load_dataset
import pandas as pd
import os


def prepare_dataset():
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")
    test_data = dataset["test"]

    questions = []
    for i in range(75):
        item = test_data[i]
        questions.append({
            "id": i + 1,
            "category": "math_reasoning",
            "english_question": item["question"],
            "correct_answer": item["answer"].split("####")[-1].strip(),
            "hindi_question": "",
            "hinglish_question": ""
        })

    os.makedirs("dataset", exist_ok=True)
    df = pd.DataFrame(questions)
    df.to_csv("dataset/questions.csv", index=False)
    print(f"Saved {len(df)} questions to dataset/questions.csv")


if __name__ == "__main__":
    prepare_dataset()