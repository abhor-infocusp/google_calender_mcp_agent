#!/usr/bin/env bash
# Evaluate untrained Qwen3-8B and Qwen3-14B (no LoRA) on the same 286-trajectory
# ART hold-out used for runs/judge_v1_qwen3_7b_20260425. Establishes the
# "do nothing" baseline before another round of judge SFT experiments.
#
# 8B fits on a MIG slice in bf16. 14B needs 4-bit (28 GiB bf16 > 24 GiB slice).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${RUN_DIR:-runs/judge_baseline_20260430}"
PYTHON=/home/abhor/miniconda3/envs/agentic/bin/python
SLICE_8B="${SLICE_8B:-MIG-dd607cdf-e8cb-531f-b478-417160625a35}"   # slice 2
SLICE_14B="${SLICE_14B:-MIG-7488039b-0c78-50bb-8112-a1ae051fc3f7}" # slice 3
NUM_SAMPLES="${NUM_SAMPLES:-286}"

mkdir -p "$RUN_DIR/eval" "$RUN_DIR/logs"

(
    LOG="$RUN_DIR/logs/baseline_qwen3_8b_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "=== Qwen3-8B baseline (bf16, no LoRA) start $(date) ==="
        CUDA_VISIBLE_DEVICES="$SLICE_8B" PYTHONPATH=src "$PYTHON" \
            scripts/eval/eval_judge_on_art.py \
            --checkpoint "" \
            --base-model Qwen/Qwen3-8B \
            --num-samples "$NUM_SAMPLES" \
            --output "$RUN_DIR/eval/art_holdout_qwen3_8b_base.json"
        echo "=== Qwen3-8B baseline done $(date) ==="
    } > "$LOG" 2>&1
) &
PID_8B=$!

(
    LOG="$RUN_DIR/logs/baseline_qwen3_14b_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "=== Qwen3-14B baseline (4-bit nf4, no LoRA) start $(date) ==="
        CUDA_VISIBLE_DEVICES="$SLICE_14B" PYTHONPATH=src "$PYTHON" \
            scripts/eval/eval_judge_on_art.py \
            --checkpoint "" \
            --base-model Qwen/Qwen3-14B \
            --load-in-4bit \
            --num-samples "$NUM_SAMPLES" \
            --output "$RUN_DIR/eval/art_holdout_qwen3_14b_base.json"
        echo "=== Qwen3-14B baseline done $(date) ==="
    } > "$LOG" 2>&1
) &
PID_14B=$!

echo "Qwen3-8B  baseline (slice $SLICE_8B):  PID $PID_8B"
echo "Qwen3-14B baseline (slice $SLICE_14B): PID $PID_14B"
echo "$PID_8B $PID_14B" > "$RUN_DIR/logs/baseline_pids.txt"
echo "Logs in:    $RUN_DIR/logs/baseline_*.log"
echo "Outputs in: $RUN_DIR/eval/art_holdout_qwen3_*_base.json"
