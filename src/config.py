# Central configuration for the entire project
# All constants live here — never hardcode values elsewhere

import os
from dotenv import load_dotenv

load_dotenv()

# Model settings
SMALL_MODEL = "llama3.2:3b"        # Local model via Ollama
LARGE_MODEL = "llama-3.1-70b-versatile"  # Groq API

# Self-consistency settings
# Why 5? Enough samples to detect disagreement without being too slow
N_SAMPLES = 5
# Why 4/5? If 4 out of 5 agree, model is confident enough to trust
CONSISTENCY_THRESHOLD = 4

# API settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OLLAMA_BASE_URL = "http://localhost:11434"

# Paths
DATASET_PATH = "dataset/questions.csv"
RESULTS_PATH = "results/raw_results.csv"