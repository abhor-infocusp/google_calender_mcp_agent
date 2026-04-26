"""Centralized path config for the calendar-agent package.

All data inputs live under `data/<family>/`:
  data/sft/    SFT training data (calendars, queries, trajectories)
  data/rl/     RL scenarios (calendars, queries)
  data/test/   Held-out eval set (49 calendars × queries)
  data/judge/  Judge-SFT train/val JSONL

Each path can be overridden by an env var of the same name (e.g.
CALENDAR_AGENT_SFT_DATA_DIR=/some/other/path) for testing or alternate datasets.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _env_or(env_name: str, default: Path) -> Path:
    v = os.environ.get(env_name)
    return Path(v) if v else default


# Top-level data dir
DATA_DIR = _env_or("CALENDAR_AGENT_DATA_DIR", PROJECT_ROOT / "data")

# Per-family roots
SFT_DATA_DIR = _env_or("CALENDAR_AGENT_SFT_DATA_DIR", DATA_DIR / "sft")
RL_DATA_DIR = _env_or("CALENDAR_AGENT_RL_DATA_DIR", DATA_DIR / "rl")
TEST_DATA_DIR = _env_or("CALENDAR_AGENT_TEST_DATA_DIR", DATA_DIR / "test")
JUDGE_DATA_DIR = _env_or("CALENDAR_AGENT_JUDGE_DATA_DIR", DATA_DIR / "judge")

# Outputs (gitignored)
SFT_OUTPUT_DIR = _env_or("CALENDAR_AGENT_SFT_OUTPUT_DIR", PROJECT_ROOT / "sft_output")

# Credentials (gitignored)
CREDENTIALS_PATH = _env_or("GOOGLE_APPLICATION_CREDENTIALS", PROJECT_ROOT / "gcloud_credentials.json")

# Conventional subpaths inside each family — derive from family roots so an
# override of CALENDAR_AGENT_SFT_DATA_DIR cascades correctly.
SFT_JSON_CALENDAR_DIR = SFT_DATA_DIR / "json_calender"
SFT_QUERY_DIR = SFT_DATA_DIR / "queries"
RL_JSON_CALENDAR_DIR = RL_DATA_DIR / "json_calender"
RL_QUERY_DIR = RL_DATA_DIR / "queries"
TEST_JSON_CALENDAR_DIR = TEST_DATA_DIR / "json_calender"
TEST_QUERY_DIR = TEST_DATA_DIR / "queries"

# Judge data has flat train.jsonl / val.jsonl (no further subdirs).
JUDGE_TRAIN_JSONL = JUDGE_DATA_DIR / "train.jsonl"
JUDGE_VAL_JSONL = JUDGE_DATA_DIR / "val.jsonl"
