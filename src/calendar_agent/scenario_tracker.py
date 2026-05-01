"""Per-scenario rolling pass-rate tracker for adaptive GRPO sampling.

Tracks how often each scenario is solved (EMA of correct/incorrect verdicts),
buckets scenarios by difficulty, and exposes sampling weights proportional to
the *expected probability of producing a non-skipped GRPO group*. With binary
rewards and group size G, that probability is

    P(non-skip | p) = 1 - p**G - (1 - p)**G

which peaks at p=0.5 and drops to zero as p→0 or p→1. Sampling proportional
to this drives the sampler toward scenarios that yield a usable advantage
signal and naturally away from saturated easy ones — keeping the GRPO skip
rate low without ad-hoc retest boosts.

References:
- AR3PO (arxiv 2509.25808): adaptive rollout + response reuse for the all-fail
  case. Multi-stage rollout and the replay buffer live in the training script
  (rl_train_adaptive.py); this module supplies the difficulty signal.
- DOTS (arxiv 2506.05316): difficulty-targeted online data selection.
- "Hard Examples Are All You Need" (arxiv 2508.14094): never permanently drop
  hard scenarios — they retain the most learning potential. Response reuse in
  rl_train_adaptive.py keeps unreachable hard scenarios in the curriculum even
  when their on-policy pass-rate hits zero.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Literal

Bucket = Literal["cold", "hard", "mid", "easy"]

DEFAULT_ALPHA = 0.3
COLD_START_OBS = 8
HARD_THRESHOLD = 0.3
EASY_THRESHOLD = 0.7
# Tiny floor so a scenario whose EMA has decayed to ~0 (or ~1) can still be
# rediscovered occasionally — multi-stage rollout + response reuse will rescue
# it. Without this floor, a scenario that hit pass_rate_ema=0 once is permanently
# dropped, which contradicts the "Hard Examples" principle above.
WEIGHT_FLOOR = 0.02


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

    def sample_weight(self, scenario_id: str, group_size: int) -> float:
        """Probability this scenario produces a non-skipped GRPO group.

        For binary rewards, a group of `group_size` rollouts is skipped iff
        all rollouts agree (all-pass or all-fail). Under an iid Bernoulli(p)
        model with p = pass_rate_ema, that probability is

            P(skip)     = p**G + (1-p)**G
            P(non-skip) = 1 - p**G - (1-p)**G

        Sampling weight ∝ P(non-skip) routes compute toward scenarios that
        actually yield a learning signal. Cold scenarios (insufficient data
        to estimate p) get uniform weight 1.0 so warmup is unbiased.
        """
        s = self.stats[scenario_id]
        if s.n_observations < self.cold_start_n:
            # Cold-start: uniform, pull every scenario through warmup.
            return 1.0
        if group_size < 2:
            # Single-rollout groups can never produce a non-trivial advantage.
            return WEIGHT_FLOOR
        p = s.pass_rate_ema
        w = 1.0 - p**group_size - (1.0 - p) ** group_size
        return max(w, WEIGHT_FLOOR)

    def sample_weights(
        self,
        group_size_fn: Callable[[str], int] | int,
    ) -> dict[str, float]:
        """Per-scenario weight map.

        group_size_fn: either a constant int (group size used for every
        scenario) or a callable scenario_id -> int (lets the caller couple
        weight to per-bucket budget). Passing a callable matters when the
        budget varies by bucket — e.g. easy=4 vs hard=8 — because P(non-skip)
        depends on G, not just p.
        """
        if callable(group_size_fn):
            return {
                sid: self.sample_weight(sid, group_size_fn(sid))
                for sid in self.stats
            }
        g = int(group_size_fn)
        return {sid: self.sample_weight(sid, g) for sid in self.stats}

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
