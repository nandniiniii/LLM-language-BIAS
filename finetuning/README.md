# Hinglish Reasoning Dataset

Hinglish (Roman-script Hindi + English) version of a GSM8K subset,
used to evaluate and fine-tune LLMs on Hinglish reasoning.

## Format (JSONL)
id, question, solution, final_answer, english_question

## How it is generated
1. Sample problems from GSM8K (fixed seed)
2. Rewrite with Groq (Llama 3.3 70B) in Hinglish, numbers unchanged
3. Auto-check: Hinglish solution must reach the same final answer
4. Manual review of a random sample

## Run
pip install -r requirements.txt
set GROQ_API_KEY=<your key>
python translate_gsm8k_hinglish.py