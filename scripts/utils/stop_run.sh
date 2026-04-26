#!/bin/bash
# Safely stop a single training run by reading its metadata.jsonl to identify
# the exact pid + script. Refuses to kill anything not matching what was
# written into the run dir's metadata. Replaces ad-hoc pgrep filters which
# have already led to one accidental kill of the wrong run.
#
# Usage: scripts/utils/stop_run.sh <RUN_DIR>
# Exit codes:
#   0 — clean stop (or already stopped)
#   1 — usage error / missing metadata
#   2 — found process but it does not match the recorded script (refuse to kill)
#   3 — kill failed

set -eu

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <RUN_DIR>" >&2
    exit 1
fi

RUN_DIR="$1"
META="$RUN_DIR/metadata.jsonl"

if [ ! -f "$META" ]; then
    echo "[stop_run] no metadata.jsonl at $META" >&2
    exit 1
fi

# Last line is the most recent process start.
LAST=$(tail -1 "$META")
PY_PID=$(echo "$LAST" | /home/abhor/miniconda3/envs/agentic/bin/python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('pid',''))")
SCRIPT=$(echo "$LAST" | /home/abhor/miniconda3/envs/agentic/bin/python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('script',''))")
TS=$(echo "$LAST" | /home/abhor/miniconda3/envs/agentic/bin/python3 -c "import sys,json; print(json.loads(sys.stdin.read()).get('ts',''))")

echo "[stop_run] reading $META"
echo "[stop_run] last entry: pid=$PY_PID  script=$(basename ${SCRIPT})  ts=$TS"

if [ -z "$PY_PID" ]; then
    echo "[stop_run] no pid in metadata; nothing to do"
    exit 0
fi

# Is the pid alive?
if ! kill -0 "$PY_PID" 2>/dev/null; then
    echo "[stop_run] pid $PY_PID not alive — already stopped"
    exit 0
fi

# Verify the live process is actually running our recorded script.
LIVE_CMD=$(ps -p "$PY_PID" -o cmd= 2>/dev/null || true)
if [ -z "$LIVE_CMD" ]; then
    echo "[stop_run] couldn't read /proc/$PY_PID/comm — refusing to kill"
    exit 2
fi
SCRIPT_BASE=$(basename "$SCRIPT")
if ! echo "$LIVE_CMD" | grep -q "$SCRIPT_BASE"; then
    echo "[stop_run] pid $PY_PID is NOT running $SCRIPT_BASE — refusing to kill" >&2
    echo "[stop_run]   live cmd: $LIVE_CMD" >&2
    exit 2
fi
echo "[stop_run] verified pid $PY_PID is $SCRIPT_BASE — proceeding"

# Find the wrapper (parent process running auto_restart.sh).
PARENT_PID=$(ps -p "$PY_PID" -o ppid= 2>/dev/null | tr -d ' ' || true)
WRAPPER_PID=""
if [ -n "$PARENT_PID" ] && [ "$PARENT_PID" -gt 1 ]; then
    PARENT_CMD=$(ps -p "$PARENT_PID" -o cmd= 2>/dev/null || true)
    if echo "$PARENT_CMD" | grep -q "auto_restart.sh"; then
        WRAPPER_PID="$PARENT_PID"
        echo "[stop_run] wrapper pid (parent of $PY_PID) = $WRAPPER_PID  ($PARENT_CMD)"
    fi
fi

# Kill wrapper FIRST so it doesn't auto-restart on our SIGTERM to the python.
if [ -n "$WRAPPER_PID" ]; then
    echo "[stop_run] killing wrapper $WRAPPER_PID (TERM, 5s grace, KILL)"
    kill -TERM "$WRAPPER_PID" 2>/dev/null || true
    sleep 1
fi

# Kill the python's process group (setsid-launched, so pgid == py_pid).
echo "[stop_run] killing pgid $PY_PID (TERM, 5s grace, KILL)"
kill -TERM -- -"$PY_PID" 2>/dev/null || true
for _ in 1 2 3 4 5; do
    if ! kill -0 "$PY_PID" 2>/dev/null; then break; fi
    sleep 1
done

if kill -0 "$PY_PID" 2>/dev/null; then
    kill -KILL -- -"$PY_PID" 2>/dev/null || true
    sleep 1
fi
[ -n "$WRAPPER_PID" ] && kill -KILL "$WRAPPER_PID" 2>/dev/null || true

# Final verification.
sleep 1
if kill -0 "$PY_PID" 2>/dev/null; then
    echo "[stop_run] FAILED: pid $PY_PID still alive after KILL" >&2
    exit 3
fi

# Check no orphan VLLM::EngineCore left from this run's process group.
ORPHANS=$(ps -e -o pid,sid,cmd --no-headers | awk -v sid="$PY_PID" '$2==sid {print $1}')
if [ -n "$ORPHANS" ]; then
    echo "[stop_run] cleaning orphan pids in session: $ORPHANS"
    echo "$ORPHANS" | xargs -r kill -KILL 2>/dev/null || true
fi

echo "[stop_run] done."
