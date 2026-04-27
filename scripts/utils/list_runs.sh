#!/bin/bash
# List all runs (with metadata.jsonl) and report which have live python procs.
# Usage: scripts/utils/list_runs.sh
set -eu

cd "$(dirname "$0")/../.."

printf "%-45s  %-7s  %-9s  %-6s  %-15s\n" "RUN_DIR" "PID" "ALIVE" "SCRIPT" "LAST_LAUNCH"
printf "%-45s  %-7s  %-9s  %-6s  %-15s\n" "$(printf '%.0s-' $(seq 1 45))" \
    "-------" "---------" "------" "---------------"

for META in runs/*/metadata.jsonl; do
    [ -f "$META" ] || continue
    RUN=$(dirname "$META")
    PID_FILE="$RUN/.run_pid"
    # Prefer .run_pid (canonical, single source of truth from telemetry).
    if [ -f "$PID_FILE" ]; then
        PID=$(sed -n '1p' "$PID_FILE")
        SCRIPT=$(sed -n '2p' "$PID_FILE" | sed 's|.*/||;s|\.py$||')
        TS="(pid-file)"
    else
        LAST=$(tail -1 "$META")
        PID=$(echo "$LAST" | /home/abhor/miniconda3/envs/agentic/bin/python3 -c \
            "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('pid',''))" 2>/dev/null || echo "")
        SCRIPT=$(echo "$LAST" | /home/abhor/miniconda3/envs/agentic/bin/python3 -c \
            "import sys,json,os.path as p; d=json.loads(sys.stdin.read()); print(p.basename(d.get('script','')).replace('.py',''))" 2>/dev/null || echo "")
        TS=$(echo "$LAST" | /home/abhor/miniconda3/envs/agentic/bin/python3 -c \
            "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('ts','')[:19])" 2>/dev/null || echo "")
    fi
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        # Confirm it's actually running the recorded script
        LIVE_CMD=$(ps -p "$PID" -o cmd= 2>/dev/null || true)
        if echo "$LIVE_CMD" | grep -q "$SCRIPT"; then
            ALIVE="ALIVE"
        else
            ALIVE="stale"
        fi
    else
        ALIVE="-"
    fi
    printf "%-45s  %-7s  %-9s  %-6s  %-15s\n" "$RUN" "$PID" "$ALIVE" "${SCRIPT:0:6}" "$TS"
done
