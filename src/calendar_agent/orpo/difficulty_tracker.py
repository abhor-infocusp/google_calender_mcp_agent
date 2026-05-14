"""Per-scenario pass-rate tracker + weighted sampler for ORPO.

Tracks each scenario's recent pass rate via EMA and exposes:
- `weight(sid)` — `P(produces_pair) = 1 − p^G − (1−p)^G`, the probability that
  a group of G rollouts at pass rate p produces *both* a correct and an
  incorrect rollout (i.e. a usable ORPO pair). Cold-start scenarios (under
  `cold_start_n` total observations) get uniform weight 1.0 to ensure warmup.
- `sample_without_replacement(n)` — draws `n` distinct scenario IDs weighted
  by `weight()`. This is the core of the AR3PO-style adaptive sampler.
- `bucket(sid)` — coarse difficulty bucket {cold, hard, mid, easy} used by
  the trainer to pick `k = 4` (easy) vs `k = 8` (hard / cold-start).

This module is a slim re-implementation of the deleted `scenario_tracker.py`.
The bucket-based sampling weight gymnastics, RETEST_BOOST, WEIGHT_FLOOR-of-0.65,
migration alert / EMA-frozen alert / forgetting alert telemetry from that
version are all removed — telemetry now lives in the trainer's per-step JSONL.

References:
- AR3PO (arxiv 2509.25808) §3.1 adaptive rollout — same intuition (more compute
  on hard prompts), implemented as adaptive sampling rather than multi-stage.
- DOTS (arxiv 2506.05316) Theorem 1 — gradient norm in GRPO is maximized at
  pass rate 0.5; sampling toward 0.5 accelerates learning. Our ORPO equivalent:
  P(produces_pair) is also maximized at p=0.5.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass
from typing import Iterable, Literal


# ── Hyperparameters (default values) ──────────────────────────────────

DEFAULT_ALPHA = 0.3
DEFAULT_COLD_START_OBS = 8
DEFAULT_HARD_THRESHOLD = 0.3
DEFAULT_EASY_THRESHOLD = 0.7
# Tiny floor so a scenario whose EMA has decayed near 0 or 1 can still be
# rediscovered occasionally — buffer rescue handles the all-fail variant,
# but easy-saturation scenarios should also stay in rotation in case the
# model regresses on them.
WEIGHT_FLOOR = 0.02


Bucket = Literal["cold", "hard", "mid", "easy"]


@dataclass
class ScenarioStats:
    scenario_id: str
    category: str
    pass_rate_ema: float = 0.5  # neutral prior
    n_observations: int = 0  # total rollouts ever scored (on-policy only)
    n_visits: int = 0  # number of times sampled as a group
    last_step: int = -1


class DifficultyTracker:
    """Per-scenario pass-rate EMA + adaptive sampler for ORPO."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        cold_start_n: int = DEFAULT_COLD_START_OBS,
        hard_threshold: float = DEFAULT_HARD_THRESHOLD,
        easy_threshold: float = DEFAULT_EASY_THRESHOLD,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0,1], got {alpha}")
        if cold_start_n < 1:
            raise ValueError(f"cold_start_n must be ≥1, got {cold_start_n}")
        if not 0.0 <= hard_threshold < easy_threshold <= 1.0:
            raise ValueError(
                f"need 0 ≤ hard_threshold ({hard_threshold}) < "
                f"easy_threshold ({easy_threshold}) ≤ 1"
            )
        self.alpha = alpha
        self.cold_start_n = cold_start_n
        self.hard_threshold = hard_threshold
        self.easy_threshold = easy_threshold
        self.stats: dict[str, ScenarioStats] = {}

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
        """Update EMA after a group of on-policy rollouts has been scored."""
        s = self.stats.get(scenario_id)
        if s is None:
            raise KeyError(f"scenario_id {scenario_id!r} not registered")
        if not rollout_correct:
            return

        for c in rollout_correct:
            s.pass_rate_ema = (1 - self.alpha) * s.pass_rate_ema + self.alpha * (
                1.0 if c else 0.0
            )
            s.n_observations += 1

        s.n_visits += 1
        s.last_step = step

        # Crash loud on corruption rather than poisoning the sampler silently.
        assert 0.0 <= s.pass_rate_ema <= 1.0, (scenario_id, s.pass_rate_ema)
        assert not math.isnan(s.pass_rate_ema), scenario_id

    # ── bucket ────────────────────────────────────────────────────────

    def bucket(self, scenario_id: str) -> Bucket:
        s = self.stats[scenario_id]
        if s.n_observations < self.cold_start_n:
            return "cold"
        if s.pass_rate_ema < self.hard_threshold:
            return "hard"
        if s.pass_rate_ema > self.easy_threshold:
            return "easy"
        return "mid"

    def k_for(self, scenario_id: str, k_easy: int = 4, k_hard: int = 8) -> int:
        """Adaptive rollout count — k_hard for hard / cold-start, k_easy otherwise."""
        b = self.bucket(scenario_id)
        return k_hard if b in ("hard", "cold") else k_easy

    # ── sampling weights ──────────────────────────────────────────────

    def weight(self, scenario_id: str, group_size: int) -> float:
        """Probability this scenario produces a non-degenerate group.

        For binary rewards, a group of `group_size` rollouts is degenerate iff
        all rollouts agree (all-pass or all-fail). Under iid Bernoulli(p) with
        p = pass_rate_ema, that's `p^G + (1−p)^G`; the non-degenerate
        probability is `1 − p^G − (1−p)^G`. Maximized at p=0.5.

        For ORPO this is also `P(produces_pair)` — degenerate groups can't
        form chosen/rejected pairs.

        Cold scenarios get uniform weight 1.0 so warmup is unbiased.
        """
        s = self.stats[scenario_id]
        if s.n_observations < self.cold_start_n:
            return 1.0
        if group_size < 2:
            return WEIGHT_FLOOR
        p = s.pass_rate_ema
        w = 1.0 - p**group_size - (1.0 - p) ** group_size
        return max(w, WEIGHT_FLOOR)

    def weights(self, k_easy: int = 4, k_hard: int = 8) -> dict[str, float]:
        """Per-scenario weight map at the k each scenario would receive."""
        return {
            sid: self.weight(sid, self.k_for(sid, k_easy=k_easy, k_hard=k_hard))
            for sid in self.stats
        }

    # ── sampling ──────────────────────────────────────────────────────

    def sample_without_replacement(
        self,
        n: int,
        *,
        k_easy: int = 4,
        k_hard: int = 8,
        rng: random.Random | None = None,
    ) -> list[str]:
        """Draw `n` distinct scenario IDs weighted by `weight()`.

        Implementation: weighted-without-replacement via the exponential trick
        — assign each item a key `−log(U) / w_i` for U ~ Uniform(0,1), take
        the n smallest keys. Equivalent to sampling proportional to `w_i`
        without replacement (Efraimidis & Spirakis 2006). O(N) per call,
        deterministic given rng.
        """
        if n < 0:
            raise ValueError(f"n must be ≥0, got {n}")
        ids = list(self.stats.keys())
        if n >= len(ids):
            return ids[:]  # ask for all → return all
        if n == 0:
            return []

        r = rng or random
        ws = [self.weight(sid, self.k_for(sid, k_easy=k_easy, k_hard=k_hard)) for sid in ids]

        # All-zero weights would NaN the keys; floor already prevents this,
        # but assert defensively in case future hyperparameter sweeps drop it.
        assert all(w > 0 for w in ws), "weight() returned non-positive; check WEIGHT_FLOOR"

        keys = [-math.log(r.random() if r.random() > 0 else 1e-300) / w for w in ws]
        # zip → sort → take n smallest keys
        order = sorted(range(len(ids)), key=lambda i: keys[i])
        return [ids[i] for i in order[:n]]

    # ── observability ─────────────────────────────────────────────────

    def bucket_counts(self) -> dict[str, int]:
        out = {"cold": 0, "hard": 0, "mid": 0, "easy": 0}
        for sid in self.stats:
            out[self.bucket(sid)] += 1
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

    # ── persistence ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        payload = {
            "alpha": self.alpha,
            "cold_start_n": self.cold_start_n,
            "hard_threshold": self.hard_threshold,
            "easy_threshold": self.easy_threshold,
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
        self.hard_threshold = payload.get("hard_threshold", self.hard_threshold)
        self.easy_threshold = payload.get("easy_threshold", self.easy_threshold)
        self.stats = {sid: ScenarioStats(**d) for sid, d in payload["stats"].items()}
