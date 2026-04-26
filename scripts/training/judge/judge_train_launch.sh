#!/usr/bin/env bash
# Bare-python launcher for the local judge SFT (Slurm broken — see CLAUDE.md).
# Pick an idle MIG slice that no RL/eval job is using; confirm with:
#   nvidia-smi -L | grep MIG
#   ps -ef | grep python
#
# UUIDs taken from scripts/eval/run_test_evals.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${JUDGE_RUN_DIR:-runs/judge_v1_qwen3_7b_20260425}"
mkdir -p "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/train_$(date +%Y%m%d_%H%M%S).log"

# Default to slice 1 (the same one used by SFT v6 evals — pick another if busy).
# Override via MIG_UUID env var.
export CUDA_VISIBLE_DEVICES="${MIG_UUID:-MIG-abbb3894-4f8c-5e33-b602-6a485436950d}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export JUDGE_RUN_DIR="$RUN_DIR"

PYTHON=/home/abhor/miniconda3/envs/agentic/bin/python

echo "Launching judge SFT"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  RUN_DIR=$RUN_DIR"
echo "  LOG=$LOG"

nohup "$PYTHON" scripts/training/judge/judge_sft_train.py \
    > "$LOG" 2>&1 &

PID=$!
echo "PID $PID -> $LOG"
echo "$PID" > "$RUN_DIR/logs/train.pid"
echo "Tail with: tail -F $LOG"
