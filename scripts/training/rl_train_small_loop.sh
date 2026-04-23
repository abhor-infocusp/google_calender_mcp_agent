#!/bin/bash
# Auto-restart wrapper for rl_train.py.
# Exit code 42 = ART queue deadlock detected by Patch G; retry.
# Any other exit code = stop (training done, bug, or user kill).
#
# Usage: nohup scripts/training/rl_train_loop.sh > runs/rl_qwen3_14b_20260420/logs/loop.log 2>&1 &

set -u
cd /home/abhor/google_calender_mcp_agent

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="runs/rl_small_qwen25_05b"
mkdir -p "${RUN_DIR}/logs/debug"

# Patch G: same 300s timeout as real RL so deadlock rates are directly comparable.
export ART_DEADLOCK_TIMEOUT_S="${ART_DEADLOCK_TIMEOUT_S:-300}"
export ART_DEADLOCK_LOG_PATH="${RUN_DIR}/logs/debug/deadlock_detected.jsonl"
# Route heartbeat + debug logs from rl_train_small.py into this run's dir.
export RL_RUN_DIR="${RUN_DIR}"

# Time-bounded experiment: stop after MAX_HOURS wall clock (default 2h).
MAX_HOURS="${MAX_HOURS:-2}"
start_epoch=$(date +%s)

MAX_RESTARTS=9999
restarts=0

cleanup_pgid() {
    # After rl_train.py exits via os._exit(42), vLLM's EngineCore
    # subprocess and the multiprocessing resource_tracker are orphaned to
    # init but RETAIN the setsid-assigned process-group ID. We kill the
    # whole group here — only our own run's children, never another run's.
    local pgid=$1
    [ -z "$pgid" ] && return
    kill -TERM -- -"$pgid" 2>/dev/null
    sleep 2
    kill -KILL -- -"$pgid" 2>/dev/null
    sleep 1
}

print_summary() {
    local total_dl=0
    [ -f "$ART_DEADLOCK_LOG_PATH" ] && total_dl=$(wc -l < "$ART_DEADLOCK_LOG_PATH")
    echo "[loop] EXIT summary: restarts=${restarts} deadlocks_logged=${total_dl}"
}

CURRENT_PGID=""
on_exit() {
    print_summary
    cleanup_pgid "$CURRENT_PGID"
}
trap on_exit EXIT

while true; do
    # 2h budget check
    now=$(date +%s)
    if [ $((now - start_epoch)) -ge $((MAX_HOURS * 3600)) ]; then
        echo "[loop] MAX_HOURS=${MAX_HOURS} reached — stopping wrapper"
        exit 0
    fi
    ts=$(date +%Y%m%d_%H%M%S)
    LOG="${RUN_DIR}/logs/train_${ts}.log"
    echo "[loop] starting rl_train_small.py (restart #${restarts}) → ${LOG}"
    # setsid puts python + vLLM in their own process group, so we can clean
    # up this run's orphans without touching any other concurrent run.
    setsid /home/abhor/miniconda3/envs/agentic/bin/python scripts/training/rl_train_small.py \
        >> "$LOG" 2>&1 &
    PY_PID=$!
    CURRENT_PGID=$PY_PID
    wait $PY_PID
    rc=$?
    echo "[loop] process exited rc=${rc} at $(date)"
    cleanup_pgid "$CURRENT_PGID"
    CURRENT_PGID=""

    if [ $rc -eq 42 ]; then
        restarts=$((restarts + 1))
        if [ $restarts -ge $MAX_RESTARTS ]; then
            echo "[loop] max restarts (${MAX_RESTARTS}) reached — giving up"
            exit 1
        fi
        echo "[loop] deadlock detected (rc=42), restarting in 10s..."
        sleep 10
        continue
    fi

    echo "[loop] exit code ${rc} is not 42 — stopping wrapper"
    exit $rc
done
