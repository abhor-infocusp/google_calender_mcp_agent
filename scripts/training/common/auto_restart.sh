#!/bin/bash
# Centralized auto-restart wrapper for any RL training script in this repo.
#
# Configured entirely via env vars (no positional args). Replaces the per-run
# wrappers (rl_train_loop.sh, rl_train_small_loop.sh, rl_stress_loop_*.sh) so
# we don't drift on isolation/timeout/setsid plumbing across scripts.
#
# Required env:
#   SCRIPT_PATH               python script to launch (e.g. scripts/training/rl/rl_train.py)
#   RUN_DIR                   output dir, e.g. runs/rl_qwen3_14b_20260420
#   CUDA_VISIBLE_DEVICES      MIG UUID (use slice_map.sh: $(slice_cuda_uuid N))
#
# Recommended env (multi-tenant isolation):
#   TASKSET_CPUS              CPU range like "0-31" — pins this run's threads.
#                             Use slice_map.sh: $(slice_cpu_range N).
#                             Empty = no pinning (for solo runs).
#   OMP_NUM_THREADS=8         Caps BLAS/OMP fan-out so concurrent runs don't fight.
#   MKL_NUM_THREADS=8
#   OPENBLAS_NUM_THREADS=8
#
# Optional env:
#   MAX_HOURS=0               Wall-clock cap. 0 = no limit.
#   MAX_RESTARTS=9999         Restart attempts on rc=42 before giving up.
#   ART_DEADLOCK_TIMEOUT_S=600         Patch G v2 per-attempt timeout.
#   ART_DEADLOCK_HARD_CEILING_S=1800   Patch G v2 hard ceiling.
#   ART_USE_THREADING_BRIDGE=0/1       1 = Patch I, else Patch G v2.
#   CHECKPOINT_KEEP_EVERY=500          Patch K — milestone checkpoints to retain.
#
# Usage:
#   source scripts/training/common/slice_map.sh
#   SLICE=0
#   SCRIPT_PATH=scripts/training/rl/rl_train.py \
#   RUN_DIR=runs/rl_qwen3_14b_20260420 \
#   CUDA_VISIBLE_DEVICES=$(slice_cuda_uuid $SLICE) \
#   TASKSET_CPUS=$(slice_cpu_range $SLICE) \
#   nohup scripts/training/common/auto_restart.sh \
#       > runs/rl_qwen3_14b_20260420/logs/loop.log 2>&1 &

set -u
cd /home/abhor/google_calender_mcp_agent

# ── Required args ──────────────────────────────────────────────────
: "${SCRIPT_PATH:?SCRIPT_PATH must be set}"
: "${RUN_DIR:?RUN_DIR must be set}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must be set (use slice_cuda_uuid)}"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "[auto_restart] SCRIPT_PATH not found: $SCRIPT_PATH" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}/logs/debug"

# ── Python defaults ───────────────────────────────────────────────
export PYTHONPATH="${PYTHONPATH:-src}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── CPU thread isolation ──────────────────────────────────────────
# Cap BLAS/OMP fan-out per process. Without this, each Python grabs `nproc`
# threads by default and concurrent trainings thrash.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ── CPU pinning via taskset ───────────────────────────────────────
TASKSET_PREFIX=""
if [ -n "${TASKSET_CPUS:-}" ]; then
    if ! command -v taskset >/dev/null 2>&1; then
        echo "[auto_restart] taskset not found; skipping CPU pinning" >&2
    else
        TASKSET_PREFIX="taskset -c ${TASKSET_CPUS}"
    fi
fi

# ── Patch G v2 deadlock detection ─────────────────────────────────
export ART_DEADLOCK_TIMEOUT_S="${ART_DEADLOCK_TIMEOUT_S:-600}"
export ART_DEADLOCK_HARD_CEILING_S="${ART_DEADLOCK_HARD_CEILING_S:-1800}"
export ART_DEADLOCK_LOG_PATH="${ART_DEADLOCK_LOG_PATH:-${RUN_DIR}/logs/debug/deadlock_detected.jsonl}"

# ── Patch K: checkpoint retention ─────────────────────────────────
export CHECKPOINT_KEEP_EVERY="${CHECKPOINT_KEEP_EVERY:-500}"

# ── Where the python script writes its debug logs / heartbeat ─────
export RL_RUN_DIR="${RUN_DIR}"

# ── Time budget + restart cap ─────────────────────────────────────
MAX_HOURS="${MAX_HOURS:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-9999}"
start_epoch=$(date +%s)
restarts=0

# ── Banner so the loop log makes the config obvious ───────────────
cat <<EOF
[auto_restart] config:
  SCRIPT_PATH              = $SCRIPT_PATH
  RUN_DIR                  = $RUN_DIR
  CUDA_VISIBLE_DEVICES     = $CUDA_VISIBLE_DEVICES
  TASKSET_CPUS             = ${TASKSET_CPUS:-<unset>}
  OMP_NUM_THREADS          = $OMP_NUM_THREADS
  MAX_HOURS                = $MAX_HOURS
  MAX_RESTARTS             = $MAX_RESTARTS
  ART_DEADLOCK_TIMEOUT_S   = $ART_DEADLOCK_TIMEOUT_S
  ART_DEADLOCK_HARD_CEILING_S = $ART_DEADLOCK_HARD_CEILING_S
  ART_USE_THREADING_BRIDGE = ${ART_USE_THREADING_BRIDGE:-0}
  CHECKPOINT_KEEP_EVERY    = $CHECKPOINT_KEEP_EVERY
EOF

# ── Process-group cleanup (carry-over from rl_train_loop.sh) ──────
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
    echo "[auto_restart] EXIT summary: restarts=${restarts} deadlock_log_lines=${total_dl}"
}

CURRENT_PGID=""
on_exit() {
    print_summary
    cleanup_pgid "$CURRENT_PGID"
}
trap on_exit EXIT

# ── Main restart loop ─────────────────────────────────────────────
while true; do
    # Wall-clock budget check.
    if [ "$MAX_HOURS" != "0" ]; then
        now=$(date +%s)
        if [ $((now - start_epoch)) -ge $((MAX_HOURS * 3600)) ]; then
            echo "[auto_restart] MAX_HOURS=${MAX_HOURS} reached — stopping wrapper"
            exit 0
        fi
    fi

    ts=$(date +%Y%m%d_%H%M%S)
    LOG="${RUN_DIR}/logs/train_${ts}.log"
    echo "[auto_restart] starting ${SCRIPT_PATH} (restart #${restarts}) → ${LOG}"

    # setsid puts python + vLLM in their own process group; cleanup_pgid then
    # kills only this run's children, not anyone else's.
    setsid ${TASKSET_PREFIX} /home/abhor/miniconda3/envs/agentic/bin/python \
        "$SCRIPT_PATH" >> "$LOG" 2>&1 &
    PY_PID=$!
    CURRENT_PGID=$PY_PID

    # Mid-run MAX_HOURS enforcer. The top-of-loop check only fires at restart
    # boundaries — if python never crashes, the budget is never re-evaluated
    # (small-RL ran 10h past its 2h cap on 2026-04-26 because of this).
    # Spawn a watcher subshell that signals the python's process group when
    # the wall-clock deadline passes. Watcher dies via the EXIT trap when the
    # main loop exits.
    WATCHER_PID=""
    if [ "$MAX_HOURS" != "0" ]; then
        REMAINING=$((MAX_HOURS * 3600 - (now - start_epoch)))
        if [ $REMAINING -le 0 ]; then REMAINING=1; fi
        (
            sleep "$REMAINING"
            if kill -0 "$PY_PID" 2>/dev/null; then
                echo "[auto_restart] MAX_HOURS=${MAX_HOURS} elapsed mid-run — terminating PGID $PY_PID"
                kill -TERM -- -"$PY_PID" 2>/dev/null
                sleep 5
                kill -KILL -- -"$PY_PID" 2>/dev/null
            fi
        ) &
        WATCHER_PID=$!
    fi

    wait $PY_PID
    rc=$?
    echo "[auto_restart] process exited rc=${rc} at $(date)"

    # Tear down the watcher (whether it fired or not).
    [ -n "$WATCHER_PID" ] && kill -TERM "$WATCHER_PID" 2>/dev/null
    [ -n "$WATCHER_PID" ] && wait "$WATCHER_PID" 2>/dev/null
    WATCHER_PID=""

    cleanup_pgid "$CURRENT_PGID"
    CURRENT_PGID=""

    if [ $rc -eq 42 ]; then
        restarts=$((restarts + 1))
        if [ $restarts -ge $MAX_RESTARTS ]; then
            echo "[auto_restart] max restarts (${MAX_RESTARTS}) reached — giving up"
            exit 1
        fi
        echo "[auto_restart] deadlock detected (rc=42), restarting in 10s..."
        sleep 10
        continue
    fi

    echo "[auto_restart] exit code ${rc} is not 42 — stopping wrapper"
    exit $rc
done
