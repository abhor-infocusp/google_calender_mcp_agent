"""Unit tests for src/calendar_agent/orpo/reuse_buffer.py."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

import pytest

from calendar_agent.orpo.reuse_buffer import (
    DEFAULT_PER_SCENARIO_CAP,
    ReuseBuffer,
)


@dataclass
class FakeTraj:
    reward: float
    metadata: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    tag: str = ""


def _t(reward: float, sid: str = "s1", tag: str = "") -> FakeTraj:
    return FakeTraj(reward=reward, metadata={"scenario_id": sid}, tag=tag)


def test_default_cap_locked():
    """Lock the published default to keep behavior consistent across runs."""
    assert DEFAULT_PER_SCENARIO_CAP == 4


def test_constructor_validates_cap():
    with pytest.raises(ValueError):
        ReuseBuffer(per_scenario_cap=0)
    with pytest.raises(ValueError):
        ReuseBuffer(per_scenario_cap=-1)


def test_only_keeps_correct():
    buf = ReuseBuffer()
    added = buf.add_correct([
        _t(1.0, "s1", "ok"),
        _t(0.0, "s1", "fail"),
        _t(0.5, "s1", "partial"),  # not exactly 1.0
        _t(1.0, "s2", "ok2"),
    ])
    assert added == 2
    assert buf.total_size() == 2
    assert buf.scenarios_covered() == 2


def test_skips_traj_without_scenario_id():
    buf = ReuseBuffer()
    orphan = FakeTraj(reward=1.0, metadata={})  # no scenario_id
    added = buf.add_correct([orphan])
    assert added == 0
    assert buf.total_size() == 0


def test_fifo_eviction_per_scenario():
    buf = ReuseBuffer(per_scenario_cap=2)
    buf.add_correct([_t(1.0, "s1", "a")])
    buf.add_correct([_t(1.0, "s1", "b")])
    buf.add_correct([_t(1.0, "s1", "c")])  # evicts "a"
    contents = {buf._buf["s1"][i].tag for i in range(2)}
    assert contents == {"b", "c"}


def test_sample_empty_returns_none():
    buf = ReuseBuffer()
    assert buf.sample("nonexistent") is None
    buf.add_correct([_t(1.0, "s1")])
    assert buf.sample("s2") is None  # different scenario
    assert buf.sample("s1") is not None


def test_sample_does_not_pop():
    """Two consecutive samples both succeed; FIFO eviction handles staleness,
    not popping. Important: the same buffered correct may be reused across
    multiple steps until it's pushed out by fresher ones."""
    buf = ReuseBuffer()
    buf.add_correct([_t(1.0, "s1")])
    a = buf.sample("s1")
    b = buf.sample("s1")
    assert a is not None and b is not None
    assert buf.total_size() == 1


def test_sample_uniform_over_contents():
    buf = ReuseBuffer(per_scenario_cap=4)
    buf.add_correct([
        _t(1.0, "s1", "a"),
        _t(1.0, "s1", "b"),
        _t(1.0, "s1", "c"),
    ])
    rng = random.Random(0)
    counts: Counter = Counter()
    for _ in range(600):
        counts[buf.sample("s1", rng=rng).tag] += 1
    # Each ~200 ± noise; 3σ window for multinomial p=1/3, n=600.
    assert set(counts) == {"a", "b", "c"}
    for tag, n in counts.items():
        assert 150 <= n <= 250, f"tag={tag} count={n}"


def test_has():
    buf = ReuseBuffer()
    assert not buf.has("s1")
    buf.add_correct([_t(1.0, "s1")])
    assert buf.has("s1")
    assert not buf.has("s2")
