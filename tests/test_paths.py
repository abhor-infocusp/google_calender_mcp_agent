import importlib
from pathlib import Path

from calendar_agent import paths


def test_paths_are_path_instances():
    for name in (
        "PROJECT_ROOT", "DATA_DIR", "SFT_DATA_DIR", "RL_DATA_DIR",
        "TEST_DATA_DIR", "JUDGE_DATA_DIR", "SFT_OUTPUT_DIR", "CREDENTIALS_PATH",
        "SFT_JSON_CALENDAR_DIR", "SFT_QUERY_DIR",
        "RL_JSON_CALENDAR_DIR", "RL_QUERY_DIR",
        "TEST_JSON_CALENDAR_DIR", "TEST_QUERY_DIR",
        "JUDGE_TRAIN_JSONL", "JUDGE_VAL_JSONL",
    ):
        assert isinstance(getattr(paths, name), Path), name


def test_default_layout():
    assert paths.DATA_DIR == paths.PROJECT_ROOT / "data"
    assert paths.SFT_DATA_DIR == paths.DATA_DIR / "sft"
    assert paths.RL_DATA_DIR == paths.DATA_DIR / "rl"
    assert paths.TEST_DATA_DIR == paths.DATA_DIR / "test"
    assert paths.JUDGE_DATA_DIR == paths.DATA_DIR / "judge"


def test_family_subpaths_cascade_from_family_root():
    assert paths.SFT_JSON_CALENDAR_DIR == paths.SFT_DATA_DIR / "json_calender"
    assert paths.SFT_QUERY_DIR == paths.SFT_DATA_DIR / "queries"
    assert paths.RL_JSON_CALENDAR_DIR == paths.RL_DATA_DIR / "json_calender"
    assert paths.TEST_JSON_CALENDAR_DIR == paths.TEST_DATA_DIR / "json_calender"


def test_judge_jsonl_paths():
    assert paths.JUDGE_TRAIN_JSONL == paths.JUDGE_DATA_DIR / "train.jsonl"
    assert paths.JUDGE_VAL_JSONL == paths.JUDGE_DATA_DIR / "val.jsonl"


def test_env_override_cascades(monkeypatch, tmp_path):
    monkeypatch.setenv("CALENDAR_AGENT_DATA_DIR", str(tmp_path))
    try:
        reloaded = importlib.reload(paths)
        assert reloaded.DATA_DIR == tmp_path
        assert reloaded.SFT_DATA_DIR == tmp_path / "sft"
        assert reloaded.SFT_JSON_CALENDAR_DIR == tmp_path / "sft" / "json_calender"
    finally:
        monkeypatch.delenv("CALENDAR_AGENT_DATA_DIR", raising=False)
        importlib.reload(paths)


def test_env_override_explicit_family_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CALENDAR_AGENT_SFT_DATA_DIR", str(tmp_path / "custom_sft"))
    try:
        reloaded = importlib.reload(paths)
        assert reloaded.SFT_DATA_DIR == tmp_path / "custom_sft"
        assert reloaded.SFT_JSON_CALENDAR_DIR == tmp_path / "custom_sft" / "json_calender"
    finally:
        monkeypatch.delenv("CALENDAR_AGENT_SFT_DATA_DIR", raising=False)
        importlib.reload(paths)
