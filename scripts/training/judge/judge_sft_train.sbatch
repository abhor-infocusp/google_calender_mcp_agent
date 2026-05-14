#!/bin/bash
#SBATCH --job-name=judge-sft
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --nodelist=azkaban

# Required env:
#   JUDGE_MODEL_NAME    HF model id (e.g. Qwen/Qwen3-14B)
#   JUDGE_RUN_DIR       run dir under runs/ (created if missing)
#   JUDGE_TRAIN_JSONL   path to train.jsonl
#   JUDGE_VAL_JSONL     path to val.jsonl
#   JUDGE_NUM_EPOCHS    int (default 3)
#   SLICE               MIG slice index 0..3 (used outside Slurm)

set -euo pipefail

: "${JUDGE_MODEL_NAME:?JUDGE_MODEL_NAME required}"
: "${JUDGE_RUN_DIR:?JUDGE_RUN_DIR required}"
: "${JUDGE_TRAIN_JSONL:?JUDGE_TRAIN_JSONL required}"
: "${JUDGE_VAL_JSONL:?JUDGE_VAL_JSONL required}"
SLICE="${SLICE:-0}"

cd /home/abhor/google_calender_mcp_agent
source scripts/training/common/slice_map.sh

mkdir -p "$JUDGE_RUN_DIR/logs" "$JUDGE_RUN_DIR/checkpoints" "$JUDGE_RUN_DIR/diagnostics"

LOG="$JUDGE_RUN_DIR/logs/train.log"
exec >>"$LOG" 2>&1

if [ -n "${SLURM_JOB_ID:-}" ]; then
    CUDA_TAG="${CUDA_VISIBLE_DEVICES:-slurm-managed}"
    CPU_TAG="slurm-managed"
else
    export CUDA_VISIBLE_DEVICES="$(slice_cuda_uuid "$SLICE")"
    CUDA_TAG="$CUDA_VISIBLE_DEVICES"
    CPU_TAG="$(slice_cpu_range "$SLICE")"
fi

echo "=== JUDGE SFT JOB ==="
echo "Job ID:   ${SLURM_JOB_ID:-bare}"
echo "Node:     ${SLURM_NODELIST:-$(hostname)}"
echo "Model:    $JUDGE_MODEL_NAME"
echo "Run dir:  $JUDGE_RUN_DIR"
echo "Train:    $JUDGE_TRAIN_JSONL"
echo "Val:      $JUDGE_VAL_JSONL"
echo "Epochs:   ${JUDGE_NUM_EPOCHS:-3}"
echo "Slice:    $SLICE  cuda=$CUDA_TAG  cpus=$CPU_TAG"
echo "Started:  $(date)"
echo "====================="

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

/home/abhor/miniconda3/envs/agentic/bin/python scripts/training/judge/judge_sft_train.py

echo "=== Finished at $(date) ==="
