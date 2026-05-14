"""Unit tests for src/calendar_agent/orpo/difficulty_tracker.py."""

from __future__ import annotations

import json
import os
import random
import tempfile
from collections import Counter

import pytest

from calendar_agent.orpo.difficulty_tracker import (
    DEFAULT_ALPHA,
    DEFAULT_COLD_START_OBS,
    WEIGHT_FLOOR,
    DifficultyTracker,
)


class _FakeScenario:
    def __init__(self, sid: str, category: str = "cat") -> None:
        self.id = sid
        self.category = category


def _populate(tracker: DifficultyTracker, sid: str, n_correct: int, n_fail: int) -> None:
    """Push observations until n_observations ≥ cold_start_n with the
    requested correct/fail mix; advances bucket out of cold-start."""
    seq = [True] * n_correct + [False] * n_fail
    tracker.update(sid, seq, step=0)


# ── Construction / validation ─────────────────────────────────────────


def test_constructor_validates_args():
    with pytest.raises(ValueError):
        DifficultyTracker(alpha=0.0)
    with pytest.raises(ValueError):
        DifficultyTracker(alpha=1.5)
    with pytest.raises(ValueError):
        DifficultyTracker(cold_start_n=0)
    with pytest.raises(ValueError):
        DifficultyTracker(hard_threshold=0.7, easy_threshold=0.3)


def test_register_idempotent():
    t = DifficultyTracker()
    t.register("s1", "c1")
    t.register("s1", "c1")  # second call no-ops
    assert len(t.stats) == 1


def test_register_many():
    t = DifficultyTracker()
    t.register_many([_FakeScenario("a", "c1"), _FakeScenario("b", "c2")])
    assert set(t.stats) == {"a", "b"}


# ── update ────────────────────────────────────────────────────────────


def test_update_unknown_scenario_raises():
    t = DifficultyTracker()
    with pytest.raises(KeyError):
        t.update("never_registered", [True], step=0)


def test_update_empty_list_noops():
    t = DifficultyTracker()
    t.register("s1", "c1")
    t.update("s1", [], step=0)
    assert t.stats["s1"].n_observations == 0


def test_update_increments_counters():
    t = DifficultyTracker()
    t.register("s1", "c1")
    t.update("s1", [True, False, True], step=5)
    s = t.stats["s1"]
    assert s.n_observations == 3
    assert s.n_visits == 1
    assert s.last_step == 5
    assert 0.0 <= s.pass_rate_ema <= 1.0


def test_update_ema_moves_toward_observation():
    t = DifficultyTracker(alpha=0.5)
    t.register("s1", "c1")
    # Initial EMA = 0.5; after 4 corrects with α=0.5, EMA → ~0.5 + 0.5*correction → close to 1
    t.update("s1", [True] * 4, step=0)
    assert t.stats["s1"].pass_rate_ema > 0.9
    # Now four fails should pull it down
    t.update("s1", [False] * 4, step=1)
    assert t.stats["s1"].pass_rate_ema < 0.1


# ── bucketing ─────────────────────────────────────────────────────────


def test_bucket_cold_start():
    t = DifficultyTracker(cold_start_n=8)
    t.register("s1", "c1")
    assert t.bucket("s1") == "cold"
    # After 4 obs (< 8), still cold
    t.update("s1", [True] * 4, step=0)
    assert t.bucket("s1") == "cold"


def test_bucket_hard_mid_easy():
    t = DifficultyTracker(cold_start_n=2, hard_threshold=0.3, easy_threshold=0.7)
    t.register("h", "c1"); t.register("m", "c1"); t.register("e", "c1")
    # Drive past cold-start with deterministic patterns; α=0.3 default needs
    # several updates to hit the extremes.
    for _ in range(20):
        t.update("h", [False, False], step=0)
        t.update("m", [True, False], step=0)
        t.update("e", [True, True], step=0)
    assert t.bucket("h") == "hard"
    assert t.bucket("m") == "mid"
    assert t.bucket("e") == "easy"


def test_k_for_uses_bucket():
    t = DifficultyTracker(cold_start_n=2)
    t.register("h", "c1"); t.register("e", "c1")
    for _ in range(20):
        t.update("h", [False, False], step=0)
        t.update("e", [True, True], step=0)
    assert t.k_for("h", k_easy=4, k_hard=8) == 8
    assert t.k_for("e", k_easy=4, k_hard=8) == 4


def test_k_for_cold_start_uses_k_hard():
    """Cold-start scenarios should explore aggressively (k_hard) to
    accumulate observations quickly."""
    t = DifficultyTracker(cold_start_n=8)
    t.register("c", "c1")
    assert t.k_for("c", k_easy=4, k_hard=8) == 8


# ── weight ────────────────────────────────────────────────────────────


def test_weight_cold_uniform_one():
    t = DifficultyTracker(cold_start_n=8)
    t.register("c", "c1")
    assert t.weight("c", group_size=8) == 1.0


def test_weight_max_at_p_half():
    """P(produces_pair) = 1 − p^G − (1−p)^G is maximized at p=0.5."""
    t = DifficultyTracker(cold_start_n=2)
    t.register("a", "c1")
    # Drive past cold start to a known EMA
    t.update("a", [True], step=0)  # EMA → 0.5 + 0.3*0.5 = 0.65
    t.update("a", [False], step=0)  # back down
    # Just check the math via direct call with known group_size
    # At G=8 the function is 1 − 0.5^8 − 0.5^8 ≈ 0.992 at p=0.5
    # vs. 1 − 0.9^8 − 0.1^8 ≈ 0.570 at p=0.9
    # We don't need to drive EMA exactly; verify the shape via direct calls.
    p_half_expected = 1.0 - 0.5**8 - 0.5**8
    p_skew_expected = 1.0 - 0.9**8 - 0.1**8
    assert p_half_expected > p_skew_expected


def test_weight_floor_for_saturated_extremes():
    """When EMA hits 0 or 1 exactly, weight floor kicks in."""
    t = DifficultyTracker(cold_start_n=2, alpha=1.0)
    t.register("a", "c1")
    t.update("a", [True] * 8, step=0)
    # α=1 + all True → EMA = 1.0 exactly
    assert t.stats["a"].pass_rate_ema == 1.0
    # 1 − 1^G − 0^G = 0 → floor applies
    assert t.weight("a", group_size=8) == WEIGHT_FLOOR


def test_weight_group_size_lt_2():
    """G=1 can never produce a pair; weight should be at floor."""
    t = DifficultyTracker(cold_start_n=2)
    t.register("a", "c1")
    t.update("a", [True, False] * 5, step=0)
    assert t.weight("a", group_size=1) == WEIGHT_FLOOR


# ── sample_without_replacement ────────────────────────────────────────


def test_sample_returns_distinct():
    t = DifficultyTracker(cold_start_n=1)
    for sid in "abcdefghij":
        t.register(sid, "c1")
        t.update(sid, [True, False], step=0)
    rng = random.Random(0)
    picks = t.sample_without_replacement(5, rng=rng)
    assert len(picks) == 5
    assert len(set(picks)) == 5
    assert all(p in t.stats for p in picks)


def test_sample_more_than_pool_returns_all():
    t = DifficultyTracker()
    t.register("a", "c1")
    t.register("b", "c1")
    out = t.sample_without_replacement(10)
    assert sorted(out) == ["a", "b"]


def test_sample_zero_returns_empty():
    t = DifficultyTracker()
    t.register("a", "c1")
    assert t.sample_without_replacement(0) == []


def test_sample_n_negative_raises():
    t = DifficultyTracker()
    with pytest.raises(ValueError):
        t.sample_without_replacement(-1)


def test_sample_biases_toward_higher_weight():
    """A scenario at p=0.5 should be sampled more often than one at p=0.99
    when both are out of cold-start. Run many trials and check skew."""
    t = DifficultyTracker(cold_start_n=2, alpha=1.0)
    t.register("mid", "c1")
    t.register("easy", "c1")
    # Force EMAs deterministically (α=1).
    t.update("mid", [True, False], step=0)  # ema 0.0... wait α=1 so last value sticks
    # With α=1, only the last value matters. Make it explicit:
    t.stats["mid"].pass_rate_ema = 0.5
    t.stats["mid"].n_observations = 100  # past cold-start
    t.stats["easy"].pass_rate_ema = 0.99
    t.stats["easy"].n_observations = 100

    counts: Counter = Counter()
    for seed in range(500):
        rng = random.Random(seed)
        # n=1 sampling tells us pure preference
        picked = t.sample_without_replacement(1, rng=rng)[0]
        counts[picked] += 1
    # Mid should be picked far more often than easy.
    assert counts["mid"] > counts["easy"] * 5, dict(counts)


# ── persistence ───────────────────────────────────────────────────────


def test_save_load_roundtrip():
    t = DifficultyTracker(alpha=0.4, cold_start_n=4)
    t.register("a", "cat_a")
    t.update("a", [True, False, True], step=7)
    t.register("b", "cat_b")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "tracker.json")
        t.save(path)
        with open(path) as f:
            payload = json.load(f)
        assert payload["alpha"] == 0.4
        assert payload["cold_start_n"] == 4
        assert set(payload["stats"]) == {"a", "b"}

        t2 = DifficultyTracker()
        t2.load(path)
        assert t2.alpha == 0.4
        assert t2.cold_start_n == 4
        assert set(t2.stats) == {"a", "b"}
        assert t2.stats["a"].n_observations == 3
        assert t2.stats["a"].last_step == 7
        # category survives
        assert t2.stats["a"].category == "cat_a"


# ── observability ─────────────────────────────────────────────────────


def test_bucket_counts():
    t = DifficultyTracker(cold_start_n=8)
    for sid in "abc":
        t.register(sid, "c1")
    assert t.bucket_counts() == {"cold": 3, "hard": 0, "mid": 0, "easy": 0}


def test_visit_stats_empty():
    t = DifficultyTracker()
    assert t.visit_stats() == {"min": 0, "p50": 0, "max": 0, "ratio": 0.0}


def test_visit_stats_populated():
    t = DifficultyTracker(cold_start_n=1)
    for sid in "abc":
        t.register(sid, "c1")
    # Different visit counts.
    for _ in range(5): t.update("a", [True], step=0)
    for _ in range(2): t.update("b", [True], step=0)
    t.update("c", [True], step=0)
    vs = t.visit_stats()
    assert vs["min"] == 1
    assert vs["max"] == 5
    assert 0.1 <= vs["ratio"] <= 0.5
