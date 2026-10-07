"""
Build a Hinglish reasoning dataset from GSM8K using the Groq API.

Colab setup:
    !pip install -q groq datasets
    Add GROQ_API_KEY in Colab Secrets (key icon on the left), then run:
    import os; from google.colab import userdata; os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")

Outputs:
    hinglish_gsm8k.jsonl  -> translated examples that passed the automatic answer check
    review_sample.csv     -> random sample to verify by hand (open in Google Sheets / Excel)
"""

import csv
import json
import os
import random
import re
import time

from datasets import load_dataset
from groq import Groq

# ----------------------------- config -----------------------------
MODEL = "llama-3.3-70b-versatile"
N_EXAMPLES = 300          # start small, check quality, then raise to 2000-5000
REVIEW_SIZE = 200         # how many rows you will manually check
OUT_FILE = "hinglish_gsm8k.jsonl"
REVIEW_FILE = "review_sample.csv"
SLEEP_SECONDS = 2         # stay under Groq free-tier rate limits
MAX_RETRIES = 3
SEED = 42

SYSTEM_PROMPT = """You rewrite English math word problems into natural Hinglish: Hindi written in Roman (Latin) script, mixed with common English words, the way young Indians actually talk and text.

Rules:
1. Roman script only. No Devanagari.
2. Keep EVERY number and every mathematical relationship exactly the same. Never change a quantity, so the final answer stays identical to the original.
3.  Change names to common Indian names in EVERY problem. Adapt foods, objects, places and events (chai, samosa, cricket, auto-rickshaw, Diwali). Use rupees ONLY if the amounts look realistic in rupees (chai, snacks, auto fare, small shop items). If they would look unrealistic (restaurant bills, cars, houses), keep the original currency. Never change any number.
4. Write the solution as short step-by-step reasoning in Hinglish, and end with a line exactly like: Final answer: <number>
5.  Write numbers and fractions as digits (3/8, 20%), not English words. Use Hindi sentence structure and verbs; keep only everyday English words (restaurant, total, tip, bus).
Return ONLY a JSON object with two string keys: "question" and "solution"."""

client = Groq(api_key=os.environ["GROQ_API_KEY"])


# ----------------------------- helpers -----------------------------
def split_gsm8k_answer(answer: str):
    """GSM8K answers look like 'reasoning... #### 42'. Remove <<calc>> tags, return (reasoning, final)."""
    body, final = answer.split("####")
    body = re.sub(r"<<.*?>>", "", body).strip()
    final = final.strip().replace(",", "")
    return body, final


def extract_final_number(text: str):
    m = re.search(r"Final answer:\s*[^\d\-]*(-?[\d,]*\.?\d+)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def translate(question: str, english_reasoning: str, final: str):
    user_msg = (
        f"English question:\n{question}\n\n"
        f"Reference solution (for your understanding of the math):\n{english_reasoning}\n\n"
        f"Correct final answer: {final}"
    )
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content)
            q, s = data.get("question"), data.get("solution")
            if isinstance(q, str) and isinstance(s, str):
                return q.strip(), s.strip()
        except Exception as e:  # network, rate limit, bad JSON
            print(f"  retry {attempt + 1}/{MAX_RETRIES}: {e}")
            time.sleep(5 * (attempt + 1))
    return None


def load_done_ids(path: str):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


# ----------------------------- main -----------------------------
def main():
    ds = load_dataset("openai/gsm8k", "main", split="train")
    indices = list(range(len(ds)))
    random.Random(SEED).shuffle(indices)
    indices = indices[:N_EXAMPLES]

    done = load_done_ids(OUT_FILE)  # lets you stop and resume
    kept, rejected = len(done), 0

    with open(OUT_FILE, "a", encoding="utf-8") as out:
        for i, idx in enumerate(indices):
            ex_id = f"gsm8k-train-{idx}"
            if ex_id in done:
                continue

            row = ds[idx]
            reasoning, final = split_gsm8k_answer(row["answer"])
            result = translate(row["question"], reasoning, final)
            if result is None:
                rejected += 1
                continue

            q_hi, sol_hi = result
            predicted = extract_final_number(sol_hi)
            # Automatic check: Hinglish solution must reach the same final answer.
            if predicted is None or abs(predicted - float(final)) > 1e-6:
                rejected += 1
                print(f"[{i}] rejected (answer mismatch): {ex_id}")
                continue

            record = {
                "id": ex_id,
                "question": q_hi,
                "solution": sol_hi,
                "final_answer": final,
                "english_question": row["question"],
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            kept += 1
            if kept % 25 == 0:
                print(f"kept {kept} | rejected {rejected} | processed {i + 1}/{len(indices)}")
            time.sleep(SLEEP_SECONDS)

    print(f"\nDone. kept={kept}, rejected={rejected}. Saved to {OUT_FILE}")
    make_review_sheet()


def make_review_sheet():
    with open(OUT_FILE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    random.Random(SEED).shuffle(rows)
    sample = rows[:REVIEW_SIZE]

    with open(REVIEW_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "english_question", "hinglish_question", "hinglish_solution",
                    "final_answer", "natural(1-5)", "math_ok(y/n)", "notes"])
        for r in sample:
            w.writerow([r["id"], r["english_question"], r["question"], r["solution"],
                        r["final_answer"], "", "", ""])
    print(f"Review sheet with {len(sample)} rows saved to {REVIEW_FILE}")


if __name__ == "__main__":
    main()