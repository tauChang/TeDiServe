# config.py
from dataclasses import dataclass

OPENAI_API_KEY = "EMPTY"
OPENAI_API_BASE = "http://localhost:8000/v1"

# Determinism-critical parameters
MODEL = None  # auto-discovered
MAX_TOKENS = 128
TEMPERATURE = 0.0
TOP_P = 1.0
N = 1
ECHO = False

# File paths
GROUND_TRUTH_FILE = "ground_truth.json"

# Number of prompts / runs
NUM_PROMPTS = 10
MULTI_RUN_TRIALS = 10
BATCH_N = 5
