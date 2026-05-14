"""Build ORPO (chosen, rejected) pairs from a step's rollouts.

For each scenario, classifies its rollouts into correct (C) and incorrect (I)
sets, then forms pairs according to four cases:

| C | I | buffer | action                                                       |
|---|---|--------|--------------------------------------------------------------|
| ≥1| ≥1| any    | emit C × I pairs (full Cartesian, no cap)                    |
| 0 | ≥1| has    | sample 1 reused correct as chosen, pair with each incorrect  |
| 0 | ≥1| empty  | skip — no chosen available                                   |
| ≥1| 0 | any    | skip — no rejected available (saturated easy)                |
| 0 | 0 | any    | skip — defensive (shouldn't happen with k ≥ 1)               |

The reused-correct pairs are tagged via `chosen.metrics["from_reuse_buffer"] = 1.0`
so downstream telemetry can split on-policy from spliced.

This module has no torch / transformers / TRL dependency — pure Python over
the trajectory protocol. See tests for full coverage of the case table.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol, TypeVar


class TrajectoryLike(Protocol):
    reward: float
    metadata: dict
    metrics: dict


T = TypeVar("T", bound=TrajectoryLike)


SkipReason = Literal["all_correct", "all_fail_no_buffer", "no_rollouts"]


@dataclass
class ScenarioPairs:
    """Result of pair construction for a single scenario this step."""

    scenario_id: str
    pairs: list[tuple]  # list of (chosen, rejected) trajectory tuples
    n_correct: int  # on-policy
    n_incorrect: int  # on-policy
    used_reuse_buffer: bool  # True if chosen came from buffer
    skip_reason: SkipReason | None  # None if pairs were emitted


def build_pairs_for_scenario(
    scenario_id: str,
    trajectories: Iterable[T],
    *,
    reuse_buffer=None,  # ReuseBuffer | None — typed loosely to avoid circular import
    rng: random.Random | None = None,
) -> ScenarioPairs:
    """Build (chosen, rejected) pairs for one scenario.

    `trajectories` is the union of all rollouts produced for this scenario
    this step (typically `k` rollouts). `reuse_buffer` is the optional
    per-scenario FIFO; if provided and on-policy yields no correct, we try
    to splice in a buffered correct.
    """
    trajs = list(trajectories)
    correct = [t for t in trajs if t.reward == 1.0]
    incorrect = [t for t in trajs if t.reward != 1.0]
    n_c, n_i = len(correct), len(incorrect)

    # Defensive case: scenario produced no rollouts at all (rollout failures).
    if n_c == 0 and n_i == 0:
        return ScenarioPairs(
            scenario_id=scenario_id,
            pairs=[],
            n_correct=0,
            n_incorrect=0,
            used_reuse_buffer=False,
            skip_reason="no_rollouts",
        )

    # All-correct: no rejected available. Saturated easy → skip.
    if n_i == 0:
        return ScenarioPairs(
            scenario_id=scenario_id,
            pairs=[],
            n_correct=n_c,
            n_incorrect=0,
            used_reuse_buffer=False,
            skip_reason="all_correct",
        )

    # All-fail on-policy: try buffer rescue.
    if n_c == 0:
        if reuse_buffer is None or not reuse_buffer.has(scenario_id):
            return ScenarioPairs(
                scenario_id=scenario_id,
                pairs=[],
                n_correct=0,
                n_incorrect=n_i,
                used_reuse_buffer=False,
                skip_reason="all_fail_no_buffer",
            )
        rescued = reuse_buffer.sample(scenario_id, rng=rng)
        # Tag the spliced trajectory so downstream can identify it.
        rescued.metrics["from_reuse_buffer"] = 1.0
        pairs = [(rescued, inc) for inc in incorrect]
        return ScenarioPairs(
            scenario_id=scenario_id,
            pairs=pairs,
            n_correct=0,
            n_incorrect=n_i,
            used_reuse_buffer=True,
            skip_reason=None,
        )

    # Mixed on-policy: full Cartesian C × I.
    pairs = [(c, i) for c in correct for i in incorrect]
    return ScenarioPairs(
        scenario_id=scenario_id,
        pairs=pairs,
        n_correct=n_c,
        n_incorrect=n_i,
        used_reuse_buffer=False,
        skip_reason=None,
    )


def build_pairs_for_step(
    rollouts_by_scenario: dict[str, list[T]],
    *,
    reuse_buffer=None,
    rng: random.Random | None = None,
) -> tuple[list[ScenarioPairs], list[tuple]]:
    """Build pairs for every scenario in this step's rollouts.

    Returns
    -------
    per_scenario : list[ScenarioPairs]
        One entry per scenario including those that were skipped (so
        telemetry can attribute skips to specific scenarios).
    flat_pairs : list[(chosen, rejected)]
        Concatenation of all scenarios' pairs, suitable for batching into
        the ORPO trainer.
    """
    per_scenario: list[ScenarioPairs] = []
    flat: list[tuple] = []
    for sid, trajs in rollouts_by_scenario.items():
        sp = build_pairs_for_scenario(
            sid, trajs, reuse_buffer=reuse_buffer, rng=rng
        )
        per_scenario.append(sp)
        flat.extend(sp.pairs)
    return per_scenario, flat
