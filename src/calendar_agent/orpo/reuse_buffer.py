"""Per-scenario FIFO buffer of correct trajectories — AR3PO line 13.

For all-fail groups under ORPO, we splice in a previously-correct trajectory
as the `chosen` and pair with on-policy fails as `rejected`. This recovers
gradient signal on scenarios that produced no correct on-policy this step
but were solvable in the past.

Key properties:
- FIFO eviction with `per_scenario_cap` (paper does not specify; we use 4).
- Uniform random sample (paper: "randomly sampled from B").
- `sample()` does NOT pop — a popular response can be reused across multiple
  steps until it's evicted by fresher correct samples.
- Trajectories without `metadata["scenario_id"]` are silently dropped on add
  (defensive against rollout failures with malformed metadata).

This buffer is in-memory only. With ~600 scenarios and an average pass rate
above zero, it warms up within ~50 training steps from a cold start.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterable, Protocol, TypeVar


DEFAULT_PER_SCENARIO_CAP = 4


class TrajectoryLike(Protocol):
    reward: float
    metadata: dict
    metrics: dict


T = TypeVar("T", bound=TrajectoryLike)


class ReuseBuffer:
    """Per-scenario FIFO buffer of correct trajectories."""

    def __init__(self, per_scenario_cap: int = DEFAULT_PER_SCENARIO_CAP) -> None:
        if per_scenario_cap < 1:
            raise ValueError(f"per_scenario_cap must be ≥1, got {per_scenario_cap}")
        self.per_scenario_cap = per_scenario_cap
        self._buf: dict[str, deque] = {}

    def add_correct(self, trajectories: Iterable[T]) -> int:
        """Push every traj with reward == 1 into its scenario's deque.
        Returns the number of trajectories actually buffered."""
        added = 0
        for t in trajectories:
            if t.reward != 1.0:
                continue
            sid = t.metadata.get("scenario_id")
            if not sid:
                continue
            buf = self._buf.setdefault(sid, deque(maxlen=self.per_scenario_cap))
            buf.append(t)
            added += 1
        return added

    def sample(self, scenario_id: str, rng: random.Random | None = None) -> T | None:
        """Uniform random correct trajectory for `scenario_id`, or None if empty."""
        buf = self._buf.get(scenario_id)
        if not buf:
            return None
        r = rng or random
        return r.choice(list(buf))

    def has(self, scenario_id: str) -> bool:
        buf = self._buf.get(scenario_id)
        return bool(buf)

    def total_size(self) -> int:
        return sum(len(b) for b in self._buf.values())

    def scenarios_covered(self) -> int:
        return sum(1 for b in self._buf.values() if b)

    def save(self, path: str) -> None:
        """Pickle the buffer to `path` (atomic via tmp+rename)."""
        import os, pickle
        snap = {sid: list(buf) for sid, buf in self._buf.items()}
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"per_scenario_cap": self.per_scenario_cap,
                         "data": snap}, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)

    def load(self, path: str) -> None:
        """Restore from `save()`. Silently no-ops if file missing."""
        import os, pickle
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            obj = pickle.load(f)
        # Trust the saved cap if present; otherwise keep the current one.
        cap = obj.get("per_scenario_cap", self.per_scenario_cap)
        self.per_scenario_cap = cap
        self._buf = {sid: deque(items, maxlen=cap)
                     for sid, items in obj["data"].items()}
