"""Per-run telemetry shared by every training script.

A trainer should call `init_telemetry(run_dir, script_path)` once at startup —
that:
  1. Writes one line to `<run_dir>/metadata.jsonl` capturing git sha, env,
     deps, sibling GPU pids, isolation knobs (taskset, OMP, etc).
  2. Starts a daemon heartbeat thread that appends `<run_dir>/logs/debug/heartbeat.jsonl`
     every 30s with the current phase + step + phase_age_s.
  3. Starts a daemon stuck-alert thread that prints `[STUCK-ALERT]` if a phase
     hasn't changed in `alert_after_s` seconds.
  4. Registers a phase-snapshot callback with `calendar_agent.art_patches` so
     Patch G can distinguish healthy inter-call waits from real hangs.

After init, trainers call `set_phase("rollouts", step=...)` at every transition
point. Heartbeat + stuck-alert + Patch G read the phase via the snapshot.

The previous in-script copies of all this (rl_train.py, rl_train_small.py)
drifted — small had no metadata, so stop_run.sh couldn't find its pid.
This module fixes that drift.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ── Phase tracker (module-level, single source of truth) ──────────────

_PHASE_LOCK = threading.Lock()
_current_phase: dict = {
    "phase": "startup",
    "step": None,
    "phase_start": time.time(),
}


def set_phase(phase: str, step: Optional[int] = None) -> None:
    """Mark the current training phase. Call at every transition point so
    heartbeats + stuck alerts + Patch G all see the latest state."""
    with _PHASE_LOCK:
        _current_phase["phase"] = phase
        _current_phase["phase_start"] = time.time()
        if step is not None:
            _current_phase["step"] = step
    # Also print so it shows up in the main train log inline.
    print(
        f"[PHASE] {phase} step={_current_phase.get('step')} "
        f"t={datetime.now().isoformat()}"
    )


def phase_snapshot() -> dict:
    """Read the current phase + how long we've been in it. Used by Patch G's
    timeout handler and any callers that need a thread-safe view."""
    with _PHASE_LOCK:
        snap = dict(_current_phase)
    return {
        "phase": snap.get("phase", "?"),
        "phase_age_s": time.time() - snap.get("phase_start", time.time()),
        "step": snap.get("step"),
    }


# ── Internal: write run metadata + start daemon threads ───────────────


def _sh(cmd: list[str], default: str = "") -> str:
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    except Exception:
        return default


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "?"


def _write_run_metadata(run_dir: str, script_path: str) -> None:
    """Append one entry to `<run_dir>/metadata.jsonl` per process start.
    Each entry captures git sha, env, deps, sibling GPU pids, isolation knobs.
    Never overwrites — the jsonl grows with each restart so a long experiment
    has a complete audit trail."""
    meta_path = os.path.join(run_dir, "metadata.jsonl")

    nv_apps = _sh([
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader",
    ])

    entry = {
        "schema_version": 2,  # v2 = isolation knobs + nvidia_smi snapshot
        "ts": datetime.now().isoformat(),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "script": script_path,
        "git_commit": _sh(["git", "rev-parse", "HEAD"], "?"),
        "git_dirty": bool(_sh(["git", "status", "--porcelain"])),
        "run_dir": run_dir,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        # Isolation knobs (set by auto_restart.sh + slice_map.sh)
        "taskset_cpus": os.environ.get("TASKSET_CPUS", ""),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        # Patches
        "art_deadlock_timeout_s": os.environ.get("ART_DEADLOCK_TIMEOUT_S", "default"),
        "art_deadlock_hard_ceiling_s": os.environ.get(
            "ART_DEADLOCK_HARD_CEILING_S", "default"
        ),
        "art_use_threading_bridge": os.environ.get("ART_USE_THREADING_BRIDGE", "0"),
        "checkpoint_keep_every": os.environ.get("CHECKPOINT_KEEP_EVERY", "default"),
        # Sibling GPU processes at our launch — empty list if we're alone.
        "nvidia_smi_compute_apps": [
            line.strip() for line in nv_apps.splitlines() if line.strip()
        ],
        "python_version": sys.version.split()[0],
        "packages": {
            pkg: _pkg_version(pkg)
            for pkg in [
                "openpipe-art", "unsloth", "trl", "transformers",
                "vllm", "torch", "peft",
            ]
        },
    }
    try:
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"[metadata] wrote run metadata → {meta_path}")
    except Exception as e:
        print(f"[metadata] failed to write: {e}")


def _heartbeat_loop(heartbeat_path: str, interval: int) -> None:
    """Append a JSONL record every `interval`s with the current phase. Runs
    in a daemon thread so it dies with the process."""
    while True:
        try:
            with _PHASE_LOCK:
                snap = dict(_current_phase)
            now = time.time()
            record = {
                "schema_version": 1,
                "ts": datetime.now().isoformat(),
                "phase": snap["phase"],
                "step": snap["step"],
                "phase_age_s": round(now - snap.get("phase_start", now), 1),
                "pid": os.getpid(),
            }
            with open(heartbeat_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            print(f"[HEARTBEAT ERROR] {e}")
        time.sleep(interval)


def _stuck_alert_loop(check_interval: int, alert_after: int) -> None:
    """LOUD warning if the current phase hasn't changed in `alert_after`s.
    Catches soft stalls Patch G's queue-bridge timeout doesn't see."""
    last_alerted_phase = None
    while True:
        try:
            with _PHASE_LOCK:
                snap = dict(_current_phase)
            age = time.time() - snap.get("phase_start", time.time())
            phase = snap.get("phase", "?")
            if age >= alert_after:
                if (phase, snap.get("phase_start")) != last_alerted_phase:
                    print(
                        f"[STUCK-ALERT] phase={phase} has been running for "
                        f"{age:.0f}s (threshold={alert_after}s). "
                        f"step={snap.get('step')}",
                        flush=True,
                    )
                    last_alerted_phase = (phase, snap.get("phase_start"))
            else:
                last_alerted_phase = None
        except Exception:
            pass
        time.sleep(check_interval)


# ── Public init ───────────────────────────────────────────────────────

_INITIALIZED = False


def init_telemetry(
    run_dir: str,
    script_path: str,
    heartbeat_interval: int = 30,
    stuck_alert_check: int = 60,
    stuck_alert_after: int = 600,
) -> None:
    """One-call setup for any training script.

    Args:
        run_dir: writes metadata.jsonl here, heartbeat under run_dir/logs/debug.
        script_path: filesystem path of the calling script (typically __file__).
        heartbeat_interval: seconds between heartbeat writes.
        stuck_alert_check: seconds between stuck-alert checks.
        stuck_alert_after: how long a phase can run before alerting.

    Idempotent — safe to call multiple times; the second call is a no-op.
    """
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    debug_dir = os.path.join(run_dir, "logs", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    heartbeat_path = os.path.join(debug_dir, "heartbeat.jsonl")

    # Register phase-getter with art_patches so Patch G can read the phase.
    try:
        from calendar_agent import art_patches as _art_patches
        _art_patches.register_phase_getter(phase_snapshot)
    except Exception:
        pass

    # Write metadata before starting threads so the very first restart's
    # metadata is captured even if the trainer crashes immediately.
    _write_run_metadata(run_dir, script_path)

    # Daemon threads die with the process; no explicit shutdown needed.
    threading.Thread(
        target=_heartbeat_loop, args=(heartbeat_path, heartbeat_interval),
        daemon=True, name="run_telemetry-heartbeat",
    ).start()
    threading.Thread(
        target=_stuck_alert_loop, args=(stuck_alert_check, stuck_alert_after),
        daemon=True, name="run_telemetry-stuck-alert",
    ).start()
