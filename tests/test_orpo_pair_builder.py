"""Unit tests for src/calendar_agent/orpo/pair_builder.py."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from calendar_agent.orpo.pair_builder import (
    ScenarioPairs,
    build_pairs_for_scenario,
    build_pairs_for_step,
)
from calendar_agent.orpo.reuse_buffer import ReuseBuffer


@dataclass
class FakeTraj:
    reward: float
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    tag: str = ""


def _t(reward: float, sid: str = "s1", tag: str = "") -> FakeTraj:
    return FakeTraj(reward=reward, metadata={"scenario_id": sid}, tag=tag)


# ── Mixed case (the happy path) ───────────────────────────────────────


def test_mixed_emits_full_cartesian():
    correct = [_t(1.0, tag=f"c{i}") for i in range(3)]
    incorrect = [_t(0.0, tag=f"i{i}") for i in range(2)]
    sp = build_pairs_for_scenario("s1", correct + incorrect)
    assert sp.skip_reason is None
    assert sp.n_correct == 3 and sp.n_incorrect == 2
    assert len(sp.pairs) == 6
    assert not sp.used_reuse_buffer
    # Verify pair shape: every pair is (correct, incorrect)
    for chosen, rejected in sp.pairs:
        assert chosen.reward == 1.0
        assert rejected.reward != 1.0


def test_mixed_one_each():
    sp = build_pairs_for_scenario("s1", [_t(1.0), _t(0.0)])
    assert sp.skip_reason is None
    assert len(sp.pairs) == 1


# ── All-correct case ──────────────────────────────────────────────────


def test_all_correct_skips_with_reason():
    sp = build_pairs_for_scenario("s1", [_t(1.0), _t(1.0), _t(1.0)])
    assert sp.skip_reason == "all_correct"
    assert sp.pairs == []
    assert sp.n_correct == 3 and sp.n_incorrect == 0
    assert not sp.used_reuse_buffer


# ── All-fail with empty buffer ────────────────────────────────────────


def test_all_fail_no_buffer_skips():
    sp = build_pairs_for_scenario("s1", [_t(0.0), _t(0.0)])
    assert sp.skip_reason == "all_fail_no_buffer"
    assert sp.pairs == []
    assert sp.n_correct == 0 and sp.n_incorrect == 2


def test_all_fail_buffer_without_this_scenario_skips():
    """Buffer has correct for scenario `other`, not for `s1` — should still skip."""
    buf = ReuseBuffer()
    buf.add_correct([_t(1.0, sid="other")])
    sp = build_pairs_for_scenario("s1", [_t(0.0), _t(0.0)], reuse_buffer=buf)
    assert sp.skip_reason == "all_fail_no_buffer"
    assert not sp.used_reuse_buffer


# ── All-fail with buffer hit (the rescue case) ────────────────────────


def test_all_fail_buffer_rescue():
    buf = ReuseBuffer()
    buf.add_correct([_t(1.0, sid="s1", tag="rescued")])
    incorrect = [_t(0.0, sid="s1", tag=f"i{i}") for i in range(3)]
    rng = random.Random(0)
    sp = build_pairs_for_scenario("s1", incorrect, reuse_buffer=buf, rng=rng)

    assert sp.skip_reason is None
    assert sp.used_reuse_buffer
    assert sp.n_correct == 0  # no on-policy correct
    assert sp.n_incorrect == 3
    assert len(sp.pairs) == 3  # one per incorrect rollout
    # Every pair has the rescued trajectory as chosen
    for chosen, rejected in sp.pairs:
        assert chosen.tag == "rescued"
        assert chosen.metrics.get("from_reuse_buffer") == 1.0
        assert rejected.reward == 0.0


def test_rescue_picks_uniformly_when_buffer_has_multiple():
    """Buffer has 3 correct candidates; over many invocations, all should
    be sampled. Doesn't need exact uniformity — just non-trivial mixing."""
    buf = ReuseBuffer(per_scenario_cap=4)
    buf.add_correct([
        _t(1.0, "s1", tag="a"),
        _t(1.0, "s1", tag="b"),
        _t(1.0, "s1", tag="c"),
    ])
    seen: set[str] = set()
    for seed in range(50):
        rng = random.Random(seed)
        sp = build_pairs_for_scenario(
            "s1", [_t(0.0, "s1")], reuse_buffer=buf, rng=rng
        )
        seen.add(sp.pairs[0][0].tag)
    assert seen == {"a", "b", "c"}


# ── Defensive: no rollouts ────────────────────────────────────────────


def test_no_rollouts_returns_skip():
    sp = build_pairs_for_scenario("s1", [])
    assert sp.skip_reason == "no_rollouts"
    assert sp.pairs == []


# ── Step-level helper ─────────────────────────────────────────────────


def test_build_pairs_for_step_aggregates():
    """Step-level helper concatenates per-scenario results."""
    rollouts = {
        "mixed":         [_t(1.0, "mixed"), _t(0.0, "mixed"), _t(1.0, "mixed")],  # 2 pairs
        "all_correct":   [_t(1.0, "all_correct"), _t(1.0, "all_correct")],         # skipped
        "all_fail":      [_t(0.0, "all_fail"), _t(0.0, "all_fail")],               # skipped (no buffer)
    }
    per_scenario, flat = build_pairs_for_step(rollouts)
    assert len(per_scenario) == 3
    skip_reasons = {sp.scenario_id: sp.skip_reason for sp in per_scenario}
    assert skip_reasons == {
        "mixed": None,
        "all_correct": "all_correct",
        "all_fail": "all_fail_no_buffer",
    }
    assert len(flat) == 2  # only mixed contributed pairs


def test_build_pairs_for_step_with_buffer_rescue():
    buf = ReuseBuffer()
    buf.add_correct([_t(1.0, "all_fail", tag="rescued")])
    rollouts = {
        "all_fail": [_t(0.0, "all_fail"), _t(0.0, "all_fail")],
    }
    per_scenario, flat = build_pairs_for_step(rollouts, reuse_buffer=buf)
    assert len(flat) == 2  # rescued correct paired with each of 2 incorrects
    assert per_scenario[0].used_reuse_buffer
    assert per_scenario[0].skip_reason is None
