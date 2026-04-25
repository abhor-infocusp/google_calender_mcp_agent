"""Per-scenario rolling pass-rate tracker for difficulty-targeted GRPO sampling.

Tracks how often each scenario is solved (EMA of correct/incorrect verdicts),
buckets scenarios by difficulty, and exposes sampling weights that bias toward
mid-difficulty (pass-rate near 0.5) — where GRPO's group-relative advantage
gives the strongest gradient signal.

References:
- AR3PO (arxiv 2509.25808): adaptive rollout + bucket allocation
- DOTS (arxiv 2506.05316): difficulty-targeted online data selection
- "Hard Examples Are All You Need" (arxiv 2508.14094): never permanently drop
  hard scenarios — they retain the most learning potential.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal

Bucket = Literal["cold", "hard", "mid", "easy"]

DEFAULT_ALPHA = 0.3
COLD_START_OBS = 8
HARD_THRESHOLD = 0.3
EASY_THRESHOLD = 0.7
RETEST_STEP_GAP = 50
RETEST_BOOST = 3.0
WEIGHT_FLOOR = 1.0 - 0.7 * 0.5  # = 0.65, never zero


@dataclass
class ScenarioStats:
    scenario_id: str
    category: str
    pass_rate_ema: float = 0.5  # neutral prior
    n_observations: int = 0     # total rollouts ever scored
    n_visits: int = 0           # number of times sampled as a group
    last_step: int = -1         # global step last sampled (-1 = never)
    last_pass_rate: float = 0.5 # for migration logging


class ScenarioTracker:
    """In-memory + JSON-persisted per-scenario stats for adaptive GRPO."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        cold_start_n: int = COLD_START_OBS,
    ) -> None:
        self.alpha = alpha
        self.cold_start_n = cold_start_n
        self.stats: dict[str, ScenarioStats] = {}
        # Per-step migration counter (reset by caller after read)
        self._migrations_this_call: dict[str, int] = {}

    # ── registration ──────────────────────────────────────────────────

    def register(self, scenario_id: str, category: str) -> None:
        if scenario_id not in self.stats:
            self.stats[scenario_id] = ScenarioStats(
                scenario_id=scenario_id, category=category
            )

    def register_many(self, scenarios: Iterable) -> None:
        for s in scenarios:
            self.register(s.id, s.category)

    # ── update ────────────────────────────────────────────────────────

    def update(
        self,
        scenario_id: str,
        rollout_correct: list[bool],
        step: int,
    ) -> None:
        """Update EMA for one scenario after a group of rollouts has been scored.

        rollout_correct: per-rollout boolean (True if judge said Correct).
        """
        s = self.stats.get(scenario_id)
        if s is None:
            raise KeyError(f"scenario_id {scenario_id!r} not registered")

        if not rollout_correct:
            return

        old_bucket = self._bucket(s)
        old_rate = s.pass_rate_ema

        # Update EMA per rollout (preserves resolution at small alpha)
        for c in rollout_correct:
            s.pass_rate_ema = (1 - self.alpha) * s.pass_rate_ema + self.alpha * (
                1.0 if c else 0.0
            )
            s.n_observations += 1

        s.n_visits += 1
        s.last_step = step
        s.last_pass_rate = old_rate

        # Invariants — crash loud if something is corrupt.
        assert 0.0 <= s.pass_rate_ema <= 1.0, (scenario_id, s.pass_rate_ema)
        assert not math.isnan(s.pass_rate_ema), scenario_id
        assert s.n_observations >= 0

        new_bucket = self._bucket(s)
        if new_bucket != old_bucket:
            key = f"{old_bucket}->{new_bucket}"
            self._migrations_this_call[key] = (
                self._migrations_this_call.get(key, 0) + 1
            )

    # ── bucket logic ──────────────────────────────────────────────────

    def _bucket(self, s: ScenarioStats) -> Bucket:
        if s.n_observations < self.cold_start_n:
            return "cold"
        if s.pass_rate_ema < HARD_THRESHOLD:
            return "hard"
        if s.pass_rate_ema > EASY_THRESHOLD:
            return "easy"
        return "mid"

    def get_bucket(self, scenario_id: str) -> Bucket:
        return self._bucket(self.stats[scenario_id])

    # ── sampling weights ──────────────────────────────────────────────

    def sample_weight(self, scenario_id: str, current_step: int) -> float:
        s = self.stats[scenario_id]
        if s.n_observations < self.cold_start_n:
            # Cold-start: uniform-ish, pull every scenario through warmup.
            return 1.0
        # Triangular peak at 0.5; floor 0.65 at extremes (0/1).
        w = 1.0 - 0.7 * abs(s.pass_rate_ema - 0.5)
        # Forced retest of graduated easy scenarios — catches forgetting.
        if (
            self._bucket(s) == "easy"
            and s.last_step >= 0
            and current_step - s.last_step > RETEST_STEP_GAP
        ):
            w *= RETEST_BOOST
        return w

    def sample_weights(self, current_step: int) -> dict[str, float]:
        return {
            sid: self.sample_weight(sid, current_step) for sid in self.stats
        }

    # ── observability ─────────────────────────────────────────────────

    def bucket_counts(self) -> dict[str, int]:
        out = {"cold": 0, "hard": 0, "mid": 0, "easy": 0}
        for s in self.stats.values():
            out[self._bucket(s)] += 1
        return out

    def visit_stats(self) -> dict[str, float]:
        visits = sorted(s.n_visits for s in self.stats.values())
        if not visits:
            return {"min": 0, "p50": 0, "max": 0, "ratio": 0.0}
        n = len(visits)
        mn, mx = visits[0], visits[-1]
        return {
            "min": mn,
            "p50": visits[n // 2],
            "max": mx,
            "ratio": (mn / mx) if mx > 0 else 0.0,
        }

    def n_observations_dist(self) -> dict[str, int]:
        obs = sorted(s.n_observations for s in self.stats.values())
        if not obs:
            return {"min": 0, "p50": 0, "max": 0}
        return {"min": obs[0], "p50": obs[len(obs) // 2], "max": obs[-1]}

    def pop_migrations(self) -> dict[str, int]:
        m = self._migrations_this_call
        self._migrations_this_call = {}
        return m

    def retest_count(self, current_step: int) -> int:
        n = 0
        for s in self.stats.values():
            if (
                self._bucket(s) == "easy"
                and s.last_step >= 0
                and current_step - s.last_step > RETEST_STEP_GAP
            ):
                n += 1
        return n

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        payload = {
            "alpha": self.alpha,
            "cold_start_n": self.cold_start_n,
            "stats": {sid: asdict(s) for sid, s in self.stats.items()},
        }
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)
        self.alpha = payload.get("alpha", self.alpha)
        self.cold_start_n = payload.get("cold_start_n", self.cold_start_n)
        self.stats = {
            sid: ScenarioStats(**d) for sid, d in payload["stats"].items()
        }
