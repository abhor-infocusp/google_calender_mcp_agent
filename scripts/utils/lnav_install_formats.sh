#!/bin/bash
# Install lnav format files for calendar-agent JSONL streams.
# Run once per machine. Idempotent.
set -eu

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
FMT_DIR="${HOME}/.lnav/formats/installed"
mkdir -p "$FMT_DIR"

for f in "$REPO_DIR/scripts/utils/lnav_formats"/*.json; do
    name=$(basename "$f")
    cp "$f" "$FMT_DIR/$name"
    echo "installed: $FMT_DIR/$name"
done

echo "Test with:"
echo "  lnav -i $REPO_DIR/scripts/utils/lnav_formats/heartbeat.json"
echo "or open everything for a run:"
echo "  scripts/utils/lnav_rl.sh runs/rl_qwen3_14b_20260420"
