#!/bin/bash
# Auto-restart wrapper for rl_train_stress.py — measures deadlock cadence
# on a tiny model with random rewards.
#
# Env vars (all optional):
#   ART_DEADLOCK_TIMEOUT_S   Patch G timeout in seconds (default 30 for stress)
#   STRESS_MAX_HOURS         auto-stop after N hours (default 2; 0 = no limit)
#   STRESS_MAX_DEADLOCKS     auto-stop after N deadlocks observed (default 0 = no limit)
#   STRESS_TAG               label appended to summary line for A/B comparison
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 nohup scripts/training/rl_stress_loop.sh \
#     > runs/rl_stress_qwen25_05b/logs/loop.log 2>&1 &

set -u
cd /home/abhor/google_calender_mcp_agent

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_DIR="runs/rl_stress_qwen3_14b"
mkdir -p "${RUN_DIR}/logs/debug"

# Patch G configuration
export ART_DEADLOCK_TIMEOUT_S="${ART_DEADLOCK_TIMEOUT_S:-300}"
export ART_DEADLOCK_LOG_PATH="${RUN_DIR}/logs/debug/deadlock_detected.jsonl"

STRESS_MAX_HOURS="${STRESS_MAX_HOURS:-2}"
STRESS_MAX_DEADLOCKS="${STRESS_MAX_DEADLOCKS:-0}"
STRESS_TAG="${STRESS_TAG:-}"

MAX_RESTARTS=500
restarts=0
start_epoch=$(date +%s)

cleanup_pgid() {
    local pgid=$1
    [ -z "$pgid" ] && return
    kill -TERM -- -"$pgid" 2>/dev/null
    sleep 2
    kill -KILL -- -"$pgid" 2>/dev/null
    sleep 1
}

print_summary() {
    local now=$(date +%s)
    local elapsed_s=$((now - start_epoch))
    local elapsed_min=$((elapsed_s / 60))
    local total_steps=0
    if ls "${RUN_DIR}/logs/train_"*.log >/dev/null 2>&1; then
        total_steps=$(grep -c "\[stress\] step=" "${RUN_DIR}/logs/train_"*.log 2>/dev/null | awk -F: '{sum+=$NF} END {print sum+0}')
    fi
    local total_deadlocks=0
    if [ -f "$ART_DEADLOCK_LOG_PATH" ]; then
        total_deadlocks=$(wc -l < "$ART_DEADLOCK_LOG_PATH")
    fi
    local dlpm=0
    [ $elapsed_min -gt 0 ] && dlpm=$(awk "BEGIN {printf \"%.2f\", $total_deadlocks / ($elapsed_min / 60)}")
    echo "[loop] SUMMARY${STRESS_TAG:+ tag=$STRESS_TAG} elapsed_min=$elapsed_min steps=$total_steps deadlocks=$total_deadlocks deadlocks_per_hour=$dlpm restarts=$restarts"
}

CURRENT_PGID=""
finalize() {
    print_summary
    cleanup_pgid "$CURRENT_PGID"
}
trap finalize EXIT

check_budget() {
    local now=$(date +%s)
    local elapsed_h=$(( (now - start_epoch) / 3600 ))
    if [ "$STRESS_MAX_HOURS" != "0" ] && [ $((now - start_epoch)) -ge $((STRESS_MAX_HOURS * 3600)) ]; then
        echo "[loop] STRESS_MAX_HOURS=$STRESS_MAX_HOURS reached — stopping"
        exit 0
    fi
    if [ "$STRESS_MAX_DEADLOCKS" != "0" ] && [ -f "$ART_DEADLOCK_LOG_PATH" ]; then
        local dl=$(wc -l < "$ART_DEADLOCK_LOG_PATH")
        if [ "$dl" -ge "$STRESS_MAX_DEADLOCKS" ]; then
            echo "[loop] STRESS_MAX_DEADLOCKS=$STRESS_MAX_DEADLOCKS reached (observed=$dl) — stopping"
            exit 0
        fi
    fi
}

while true; do
    check_budget
    ts=$(date +%Y%m%d_%H%M%S)
    LOG="${RUN_DIR}/logs/train_${ts}.log"
    echo "[loop] starting rl_train_stress.py (restart #${restarts}) timeout=${ART_DEADLOCK_TIMEOUT_S}s → ${LOG}"
    setsid /home/abhor/miniconda3/envs/agentic/bin/python scripts/training/rl_train_stress_14b.py \
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
