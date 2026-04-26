#!/bin/bash
# Open lnav on a run dir with all our JSONL streams loaded.
# Usage: scripts/utils/lnav_rl.sh <RUN_DIR> [-c '<sql>'] [-c '<sql>'] ...
#        scripts/utils/lnav_rl.sh <RUN_DIR> --exec '<sql>'         # alias for -c
#
# Examples:
#   scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420                       # interactive TUI
#   scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420 \
#     -c ';SELECT count(*), event FROM _calendar_agent_deadlock GROUP BY event'

set -eu

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <RUN_DIR> [-c '<query>']..." >&2
    echo "Example: $0 runs/rl_qwen3_14b_20260420" >&2
    exit 1
fi

RUN_DIR="$1"
shift || true

# Translate --exec to lnav's -c, and run non-interactively (-n) when any -c is passed.
ARGS=()
NON_INTERACTIVE=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --exec) ARGS+=("-c" "$2"); NON_INTERACTIVE="-n"; shift 2 ;;
        -c)     ARGS+=("-c" "$2"); NON_INTERACTIVE="-n"; shift 2 ;;
        *)      ARGS+=("$1"); shift ;;
    esac
done

if [ ! -d "$RUN_DIR" ]; then
    echo "RUN_DIR not found: $RUN_DIR" >&2
    exit 1
fi

LNAV="${LNAV_BIN:-$HOME/.local/bin/lnav}"
[ -x "$LNAV" ] || LNAV="$(command -v lnav)" || {
    echo "lnav binary not found. Run scripts/utils/lnav_install_formats.sh first" >&2
    exit 1
}

# Files to load: deadlock + heartbeat + metadata + train log STEP SUMMARY (only).
FILES=()
[ -f "$RUN_DIR/logs/debug/heartbeat.jsonl" ] && FILES+=("$RUN_DIR/logs/debug/heartbeat.jsonl")
[ -f "$RUN_DIR/logs/debug/deadlock_detected.jsonl" ] && FILES+=("$RUN_DIR/logs/debug/deadlock_detected.jsonl")
[ -f "$RUN_DIR/metadata.jsonl" ] && FILES+=("$RUN_DIR/metadata.jsonl")

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No JSONL streams found under $RUN_DIR" >&2
    exit 1
fi

exec "$LNAV" $NON_INTERACTIVE "${ARGS[@]}" "${FILES[@]}"
