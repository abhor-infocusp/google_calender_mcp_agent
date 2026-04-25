#!/bin/bash
# Auto-restart wrapper for rl_train_adaptive.py.
# Mirror of rl_train_loop.sh but points at the adaptive script and a
# separate RUN_DIR so it can run in parallel with the vanilla GRPO run.
#
# Exit code 42 = ART queue deadlock detected by Patch G; retry.
# Any other exit code = stop (training done, bug, or user kill).
#
# Usage:
#   nohup scripts/training/rl_train_adaptive_loop.sh \
#     > runs/rl_adaptive_qwen3_14b_20260424/logs/loop.log 2>&1 &

set -u
cd /home/abhor/google_calender_mcp_agent

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="runs/rl_adaptive_qwen3_14b_20260424"
mkdir -p "${RUN_DIR}/logs/debug"
export RL_RUN_DIR="${RUN_DIR}"

# Patch G v2 — same defaults as the vanilla loop.
export ART_DEADLOCK_TIMEOUT_S="${ART_DEADLOCK_TIMEOUT_S:-600}"
export ART_DEADLOCK_HARD_CEILING_S="${ART_DEADLOCK_HARD_CEILING_S:-1800}"
export ART_DEADLOCK_LOG_PATH="${RUN_DIR}/logs/debug/deadlock_detected.jsonl"

MAX_RESTARTS=9999
restarts=0

cleanup_pgid() {
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
    ts=$(date +%Y%m%d_%H%M%S)
    LOG="${RUN_DIR}/logs/train_${ts}.log"
    echo "[loop] starting rl_train_adaptive.py (restart #${restarts}) → ${LOG}"
    setsid /home/abhor/miniconda3/envs/agentic/bin/python scripts/training/rl_train_adaptive.py \
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
