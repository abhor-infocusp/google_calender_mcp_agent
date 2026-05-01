"""Tests for ScenarioTracker — the difficulty signal behind adaptive GRPO.

Focus: the new sample_weight semantics. Weight at scenario id `s` with group
size G must equal P(non-skip group) = 1 - p**G - (1-p)**G under iid Bernoulli(p)
rollouts where p = pass_rate_ema. This ties sampling probability directly to
the chance of producing a non-zero advantage signal, replacing the older
ad-hoc "triangle weight + retest boost" formulation.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from calendar_agent.scenario_tracker import (
    EASY_THRESHOLD,
    HARD_THRESHOLD,
    WEIGHT_FLOOR,
    ScenarioTracker,
)


def _tracker_with(stats: dict[str, tuple[float, int]]) -> ScenarioTracker:
    """stats: {scenario_id: (pass_rate_ema, n_observations)}."""
    t = ScenarioTracker(cold_start_n=8)
    for sid, (p, n_obs) in stats.items():
        t.register(sid, category="test")
        s = t.stats[sid]
        s.pass_rate_ema = p
        s.n_observations = n_obs
    return t


def test_weight_equals_non_skip_probability():
    t = _tracker_with({"a": (0.5, 100), "b": (0.1, 100), "c": (0.9, 100)})
    G = 8
    for sid, p in [("a", 0.5), ("b", 0.1), ("c", 0.9)]:
        expected = 1.0 - p**G - (1.0 - p) ** G
        assert t.sample_weight(sid, G) == pytest.approx(expected, rel=1e-9)


def test_weight_peaks_at_p_half():
    t = _tracker_with({sid: (p, 100) for sid, p in [
        ("p10", 0.1), ("p30", 0.3), ("p50", 0.5), ("p70", 0.7), ("p90", 0.9)
    ]})
    G = 8
    weights = {sid: t.sample_weight(sid, G) for sid in t.stats}
    assert weights["p50"] == max(weights.values()), weights
    # Symmetric around 0.5 (within float)
    assert weights["p10"] == pytest.approx(weights["p90"], abs=1e-9)
    assert weights["p30"] == pytest.approx(weights["p70"], abs=1e-9)


def test_saturated_scenarios_get_floor_not_zero():
    """A scenario whose EMA collapsed to ~0 should still be sampleable so
    multi-stage rollout / response reuse can rescue it."""
    t = _tracker_with({"dead": (0.0, 100), "perfect": (1.0, 100)})
    G = 8
    assert t.sample_weight("dead", G) == pytest.approx(WEIGHT_FLOOR)
    assert t.sample_weight("perfect", G) == pytest.approx(WEIGHT_FLOOR)


def test_cold_scenarios_get_uniform_weight():
    t = _tracker_with({"cold1": (0.5, 0), "cold2": (0.95, 4), "hot": (0.5, 100)})
    G = 8
    assert t.sample_weight("cold1", G) == 1.0
    assert t.sample_weight("cold2", G) == 1.0
    # Mid-difficulty scenario at p=0.5 with G=8 has w = 1 - 2*0.5^8 ≈ 0.992
    assert t.sample_weight("hot", G) < 1.0
    assert t.sample_weight("hot", G) > 0.99


def test_weight_monotonically_increasing_in_group_size():
    """P(non-skip) = 1 - p^G - (1-p)^G is strictly increasing in G for any
    p ∈ (0, 1) — more rollouts make all-same outcomes less likely. This is
    why we couple weight to G: when easy scenarios get a smaller budget, they
    correctly receive a smaller weight than mid scenarios at the same p."""
    t = _tracker_with({"x": (0.85, 100), "y": (0.50, 100)})
    for sid in ("x", "y"):
        prev = 0.0
        for G in (2, 4, 8, 16, 32):
            w = t.sample_weight(sid, G)
            assert w >= prev, (sid, G, w, prev)
            prev = w


def test_easy_with_small_budget_underweighted_vs_mid_with_large_budget():
    """The configured policy is easy=4 rollouts, mid=8. At p=0.85 (easy)
    vs p=0.5 (mid), the easy scenario must receive lower weight than the mid
    scenario — regardless of how many easy scenarios exist in the pool."""
    t = _tracker_with({"easy": (0.85, 100), "mid": (0.50, 100)})
    w_easy = t.sample_weight("easy", 4)
    w_mid = t.sample_weight("mid", 8)
    assert w_easy < w_mid, (w_easy, w_mid)


def test_bucket_classification():
    t = _tracker_with({
        "h": (HARD_THRESHOLD - 0.05, 100),
        "m": (0.5, 100),
        "e": (EASY_THRESHOLD + 0.05, 100),
        "c": (0.5, 1),
    })
    assert t.get_bucket("h") == "hard"
    assert t.get_bucket("m") == "mid"
    assert t.get_bucket("e") == "easy"
    assert t.get_bucket("c") == "cold"


def test_sample_weights_with_callable_budget():
    """When G varies per scenario, sample_weights must use the callable
    budget so each scenario's weight reflects the group size it would actually
    receive."""
    t = _tracker_with({"easy": (0.85, 100), "hard": (0.15, 100)})

    def budget(sid: str) -> int:
        return 4 if sid == "easy" else 8

    weights = t.sample_weights(budget)
    assert weights["easy"] == pytest.approx(t.sample_weight("easy", 4))
    assert weights["hard"] == pytest.approx(t.sample_weight("hard", 8))


def test_sample_weights_with_constant_budget():
    t = _tracker_with({"a": (0.5, 100), "b": (0.3, 100)})
    weights = t.sample_weights(8)
    assert weights["a"] == pytest.approx(t.sample_weight("a", 8))
    assert weights["b"] == pytest.approx(t.sample_weight("b", 8))


def test_update_advances_ema_and_visit_count():
    t = _tracker_with({"s": (0.5, 0)})  # cold-start
    t.update("s", [True, False, True, True], step=10)
    s = t.stats["s"]
    assert s.n_observations == 4
    assert s.n_visits == 1
    assert s.last_step == 10
    # pass rate moved toward 0.75 (3/4 correct)
    assert 0.5 < s.pass_rate_ema < 1.0


def test_save_load_roundtrip(tmp_path):
    t = _tracker_with({"a": (0.4, 50), "b": (0.7, 50)})
    p = tmp_path / "tracker.json"
    t.save(str(p))

    t2 = ScenarioTracker(cold_start_n=8)
    t2.load(str(p))
    assert t2.stats["a"].pass_rate_ema == pytest.approx(0.4)
    assert t2.stats["b"].pass_rate_ema == pytest.approx(0.7)
    # Sampling weights match after reload.
    assert t.sample_weight("a", 8) == pytest.approx(t2.sample_weight("a", 8))


def test_simulation_sampling_distribution_avoids_saturated_easy():
    """End-to-end: with the bucket distribution we observed in production
    (113H/116M/358E/35C with current EMAs), the new sampler should put more
    selection mass on hard+mid than on easy. Under the old retest-boosted
    formula, easy received >70%.
    """
    import random
    random.seed(0)
    t = ScenarioTracker(cold_start_n=8)
    # Generate a synthetic pool resembling production state.
    def add(prefix, n, p):
        for i in range(n):
            sid = f"{prefix}{i}"
            t.register(sid, category="test")
            t.stats[sid].pass_rate_ema = p
            t.stats[sid].n_observations = 100  # past cold start
    add("H", 113, 0.10)   # hard
    add("M", 116, 0.50)   # mid
    add("E", 358, 0.92)   # very easy (saturated)
    # cold scenarios skipped — they get uniform weight regardless

    def budget(sid: str) -> int:
        s = t.stats[sid]
        if s.pass_rate_ema > 0.7:
            return 4
        return 8

    weights = t.sample_weights(budget)
    ids = list(weights.keys())
    ws = [weights[i] for i in ids]
    n_samples = 5000
    picked = random.choices(ids, weights=ws, k=n_samples)
    counts = {"H": 0, "M": 0, "E": 0}
    for s in picked:
        counts[s[0]] += 1
    shares = {k: v / n_samples for k, v in counts.items()}
    # Mid should dominate (peak weight × moderate count).
    assert shares["M"] > shares["E"], shares
    # Easy share must collapse vs the broken formula's 70%+.
    assert shares["E"] < 0.40, shares
    # Hard gets meaningful share (was ~8% under old formula).
    assert shares["H"] > 0.10, shares
