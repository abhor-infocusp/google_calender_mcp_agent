"""End-to-end test of the RL → judge integration.

Loads rl_train_adaptive.evaluate_trajectory via importlib (avoids running
training) and exercises both paths:

  A. Happy path: live judge → returns "Correct" or "Incorrect".
  B. Hard-fail: JUDGE_URL pointed at a dead port → SystemExit(43).

Run with the judge service up:
    PYTHONPATH=src python scripts/eval/judge_rl_integration_smoke.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "scripts/training/rl/rl_train_adaptive.py"


def load_trainer():
    spec = importlib.util.spec_from_file_location("rl_train_adaptive", TRAINER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rl_train_adaptive"] = mod
    spec.loader.exec_module(mod)
    return mod


# A small synthetic scenario equivalent to what rollout() would assemble.
def _evt(s, e, summary, attendees=None, attending="ACCEPT"):
    return {"start": f"2026-05-02 {s}:00", "end": f"2026-05-02 {e}:00",
            "summary": summary, "attendees": attendees or [], "attending": attending}


SCENARIO = dict(
    query="Book me a haircut appointment for Saturday at 10 AM.",
    final_output="Scheduled for Saturday at 10:00.",
    expected="A new event is created on Saturday at 10:00.",
    before_days={"Saturday": []},
    after_days={"Saturday": [_evt("10:00", "10:30", "Haircut Appointment")]},
    category="Schedule a Single Event",
    scenario_id="smoke-test-1",
)


async def test_happy(mod):
    print("\n[A] happy path — live judge")
    v = await mod.evaluate_trajectory(**SCENARIO)
    print(f"    verdict: {v}")
    assert v in ("Correct", "Incorrect"), f"unexpected verdict: {v!r}"
    print("    PASS")


async def test_hard_fail(mod):
    print("\n[B] hard-fail path — JUDGE_URL points at a dead port")
    # Force the client to rebuild against the dead URL.
    from calendar_agent.judge import client as judge_client
    await judge_client.aclose()
    os.environ["JUDGE_URL"] = "http://127.0.0.1:9"
    judge_client.JUDGE_URL = "http://127.0.0.1:9"
    judge_client.JUDGE_TIMEOUT = 2.0
    try:
        await mod.evaluate_trajectory(**SCENARIO)
    except SystemExit as e:
        print(f"    SystemExit({e.code}) — {'PASS' if e.code == 43 else 'FAIL: expected 43'}")
        return
    print("    FAIL: expected SystemExit(43), got normal return")


async def main():
    print("loading rl_train_adaptive (this triggers art/torch/art_patches imports) …")
    mod = load_trainer()
    await test_happy(mod)
    await test_hard_fail(mod)


if __name__ == "__main__":
    asyncio.run(main())
