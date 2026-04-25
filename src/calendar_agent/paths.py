import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = PROJECT_ROOT / "data"
SFT_DATA_DIR = PROJECT_ROOT / "sft_data"
RL_DATA_DIR = PROJECT_ROOT / "rl_data"
SFT_OUTPUT_DIR = PROJECT_ROOT / "sft_output"
CREDENTIALS_PATH = PROJECT_ROOT / "gcloud_credentials.json"

SFT_JSON_CALENDAR_DIR = SFT_DATA_DIR / "json_calender"
SFT_QUERY_DIR = SFT_DATA_DIR / "queries"
RL_JSON_CALENDAR_DIR = RL_DATA_DIR / "json_calender"
RL_QUERY_DIR = RL_DATA_DIR / "queries"

TEST_DATA_DIR = PROJECT_ROOT / "test_data"
TEST_JSON_CALENDAR_DIR = TEST_DATA_DIR / "json_calender"
TEST_QUERY_DIR = TEST_DATA_DIR / "queries"
