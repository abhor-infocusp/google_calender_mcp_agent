#!/usr/bin/env python3
"""Minimal reproducer for ART 0.5.17's asyncio.Queue + nest_asyncio deadlock.

Closely mirrors art/unsloth/service.py's pattern, plus optional stressors:
  - background async tasks (simulates vLLM health monitor, request handlers)
  - GIL-yielding C work (numpy ops) inside trainer.train to mimic torch
  - generator-based async-for chain (matches _train_shared's "async for r in
    process_train_batch")
  - gradient accumulation inner loop (2 inner iters per outer, warmup)

Usage:
    python tests/repro_art_deadlock.py --trials 5 --steps 10000
    python tests/repro_art_deadlock.py --stress                # all stressors on
    python tests/repro_art_deadlock.py --stress --bg-tasks 20  # heavy contention
"""

import argparse
import asyncio
import random
import threading
import time

import nest_asyncio
import numpy as np

nest_asyncio.apply()


async def run_trial(
    num_steps: int,
    step_timeout: float,
    jitter: bool,
    bg_tasks: int = 0,
    heavy_compute: bool = False,
    grad_accum: int = 1,
):
    inputs_queue: asyncio.Queue = asyncio.Queue()
    results_queue: asyncio.Queue = asyncio.Queue()

    state = {"completed": 0, "deadlock": False, "last_progress_ts": time.time(), "stop": False}
    stop_watchdog = threading.Event()

    def watchdog():
        while not stop_watchdog.is_set():
            time.sleep(0.1)
            if time.time() - state["last_progress_ts"] > step_timeout:
                if state["completed"] < num_steps and not state["stop"]:
                    state["deadlock"] = True
                return

    wd = threading.Thread(target=watchdog, daemon=True)
    wd.start()

    def _async_prepare_inputs():
        """Exact ART pattern, no timeout in inner run."""
        async def get_inputs():
            return await inputs_queue.get()
        return asyncio.run(get_inputs())

    def fake_log():
        results_queue.put_nowait({"loss": 0.1})

    def sync_trainer_train():
        for _ in range(num_steps):
            if state["deadlock"] or state["stop"]:
                return
            for _ in range(grad_accum):
                inputs = _async_prepare_inputs()
                if state["deadlock"]:
                    return
                if heavy_compute:
                    # Numpy releases the GIL during matmul — mimics torch kernels
                    a = np.random.randn(256, 256).astype(np.float32)
                    b = np.random.randn(256, 256).astype(np.float32)
                    _ = a @ b
                elif jitter:
                    time.sleep(random.uniform(0, 0.001))
            fake_log()
            state["completed"] += 1
            state["last_progress_ts"] = time.time()

    async def train_async():
        sync_trainer_train()

    train_task = asyncio.create_task(train_async())

    async def producer_gen():
        """Async generator — mirrors process_train_batch exactly."""
        for step in range(num_steps):
            if state["deadlock"] or state["stop"]:
                return
            for _ in range(grad_accum):
                inputs_queue.put_nowait({"step": step})
                get_task = asyncio.create_task(results_queue.get())
                done, _pending = await asyncio.wait(
                    [get_task, train_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if train_task in done:
                    if not get_task.done():
                        get_task.cancel()
                    return
                results_queue.task_done()
                yield {"step": step}

    async def outer_consumer():
        """Mirrors `async for result in process_train_batch(...)` in _train_shared."""
        async for _result in producer_gen():
            if jitter:
                await asyncio.sleep(0)  # yield
            if state["deadlock"] or state["stop"]:
                return

    # Optional: background async tasks to add scheduler churn
    async def background_noise(idx: int):
        while not state["stop"] and not state["deadlock"]:
            await asyncio.sleep(random.uniform(0.001, 0.01))
            # Do some async work
            fut = asyncio.get_event_loop().create_future()
            asyncio.get_event_loop().call_soon(fut.set_result, idx)
            await fut

    bg = [asyncio.create_task(background_noise(i)) for i in range(bg_tasks)]

    try:
        consumer_task = asyncio.create_task(outer_consumer())
        while not consumer_task.done() and not train_task.done():
            if state["deadlock"]:
                break
            await asyncio.sleep(0.05)
        state["stop"] = True
        for t in bg:
            t.cancel()
        try:
            await asyncio.wait_for(
                asyncio.gather(consumer_task, train_task, *bg, return_exceptions=True),
                timeout=step_timeout + 1,
            )
        except asyncio.TimeoutError:
            pass
    finally:
        stop_watchdog.set()
        wd.join(timeout=1.0)

    return state["completed"], state["deadlock"]


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--step-timeout", type=float, default=3.0)
    parser.add_argument("--no-jitter", action="store_true")
    parser.add_argument("--bg-tasks", type=int, default=0)
    parser.add_argument("--heavy-compute", action="store_true")
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--stress", action="store_true",
                        help="Enable all stressors: 10 bg tasks + heavy compute + grad_accum 2")
    args = parser.parse_args()

    if args.stress:
        args.bg_tasks = max(args.bg_tasks, 10)
        args.heavy_compute = True
        args.grad_accum = max(args.grad_accum, 2)

    print(
        f"Config: trials={args.trials} steps={args.steps} "
        f"step_timeout={args.step_timeout}s jitter={not args.no_jitter} "
        f"bg_tasks={args.bg_tasks} heavy_compute={args.heavy_compute} "
        f"grad_accum={args.grad_accum}"
    )
    print()

    total_completed = 0
    total_attempted = 0
    trials_deadlocked = 0
    for i in range(args.trials):
        t0 = time.time()
        completed, deadlocked = await run_trial(
            args.steps,
            args.step_timeout,
            jitter=not args.no_jitter,
            bg_tasks=args.bg_tasks,
            heavy_compute=args.heavy_compute,
            grad_accum=args.grad_accum,
        )
        dt = time.time() - t0
        marker = "DEADLOCK" if deadlocked else "OK      "
        rate = completed / dt if dt > 0 else 0
        print(
            f"Trial {i+1:3d}: {marker}  completed={completed:5d}/{args.steps}  "
            f"wall={dt:5.1f}s  rate={rate:6.0f}/s"
        )
        total_completed += completed
        total_attempted += args.steps
        if deadlocked:
            trials_deadlocked += 1

    print()
    print(f"Trials deadlocked: {trials_deadlocked}/{args.trials}")
    print(f"Steps completed: {total_completed}/{total_attempted}")
    if trials_deadlocked > 0 and total_completed > 0:
        print(f"Estimated mean-steps-between-deadlocks: ~{total_completed / trials_deadlocked:.0f}")


if __name__ == "__main__":
    asyncio.run(main())
