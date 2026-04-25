#!/usr/bin/env bash
# Evaluate the 3 judge SFT epoch checkpoints on ART trajectories.
# Slice 2 chains ckpt-990 → ckpt-1980; slice 3 runs ckpt-2970 in parallel.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="${JUDGE_RUN_DIR:-runs/judge_v1_qwen3_7b_20260425}"
PYTHON=/home/abhor/miniconda3/envs/agentic/bin/python
SLICE_2=MIG-dd607cdf-e8cb-531f-b478-417160625a35
SLICE_3=MIG-7488039b-0c78-50bb-8112-a1ae051fc3f7
NUM_SAMPLES="${NUM_SAMPLES:-286}"

mkdir -p "$RUN_DIR/eval" "$RUN_DIR/logs"

run_eval() {
    local slice="$1" ckpt_name="$2" out_suffix="$3"
    CUDA_VISIBLE_DEVICES="$slice" PYTHONPATH=src "$PYTHON" \
        scripts/eval/eval_judge_on_art.py \
        --checkpoint "$RUN_DIR/checkpoints/$ckpt_name" \
        --num-samples "$NUM_SAMPLES" \
        --output "$RUN_DIR/eval/art_holdout_${out_suffix}.json"
}

# Slice 2 chain: epoch 1 then epoch 2
(
    LOG="$RUN_DIR/logs/eval_chain_slice2_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "=== chain start $(date) ==="
        echo "[slice 2] eval ckpt-990"
        run_eval "$SLICE_2" "checkpoint-990"  "ckpt990"
        echo "[slice 2] eval ckpt-1980"
        run_eval "$SLICE_2" "checkpoint-1980" "ckpt1980"
        echo "=== chain done $(date) ==="
    } > "$LOG" 2>&1
) &
PID_SLICE2=$!

# Slice 3 standalone: ckpt-2970 (= epoch 3 = final)
(
    LOG="$RUN_DIR/logs/eval_slice3_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "=== eval start $(date) ==="
        echo "[slice 3] eval ckpt-2970"
        run_eval "$SLICE_3" "checkpoint-2970" "ckpt2970"
        echo "=== eval done $(date) ==="
    } > "$LOG" 2>&1
) &
PID_SLICE3=$!

echo "Slice 2 chain (ckpt-990 → ckpt-1980): PID $PID_SLICE2"
echo "Slice 3 standalone (ckpt-2970):       PID $PID_SLICE3"
echo "$PID_SLICE2 $PID_SLICE3" > "$RUN_DIR/logs/eval_pids.txt"
echo "Logs in: $RUN_DIR/logs/eval_*"
echo "Outputs in: $RUN_DIR/eval/art_holdout_*.json"
