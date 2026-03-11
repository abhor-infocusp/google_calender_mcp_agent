import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SFT_DATA_DIR = PROJECT_ROOT / "sft_data"
RL_DATA_DIR = PROJECT_ROOT / "rl_data"
SFT_OUTPUT_DIR = PROJECT_ROOT / "sft_output"
CREDENTIALS_PATH = PROJECT_ROOT / "gcloud_credentials.json"
