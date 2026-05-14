#!/bin/bash
#SBATCH --job-name=sft-train
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --nodelist=azkaban

# Required env:
#   SFT_MODEL_NAME  HF model id (e.g. Qwen/Qwen3-8B)
#   SFT_RUN_DIR     run dir under runs/ (created if missing)
#   SLICE           MIG slice index 0..3 (used outside Slurm; under Slurm cgroup pins us)

set -euo pipefail

: "${SFT_MODEL_NAME:?SFT_MODEL_NAME required}"
: "${SFT_RUN_DIR:?SFT_RUN_DIR required}"
SLICE="${SLICE:-0}"

cd /home/abhor/google_calender_mcp_agent
source scripts/training/common/slice_map.sh

mkdir -p "$SFT_RUN_DIR/logs" "$SFT_RUN_DIR/checkpoints" "$SFT_RUN_DIR/diagnostics"

LOG="$SFT_RUN_DIR/logs/train.log"
exec >>"$LOG" 2>&1

if [ -n "${SLURM_JOB_ID:-}" ]; then
    CUDA_TAG="${CUDA_VISIBLE_DEVICES:-slurm-managed}"
    CPU_TAG="slurm-managed"
else
    export CUDA_VISIBLE_DEVICES="$(slice_cuda_uuid "$SLICE")"
    CUDA_TAG="$CUDA_VISIBLE_DEVICES"
    CPU_TAG="$(slice_cpu_range "$SLICE")"
fi

echo "=== SFT JOB ==="
echo "Job ID:   ${SLURM_JOB_ID:-bare}"
echo "Node:     ${SLURM_NODELIST:-$(hostname)}"
echo "Model:    $SFT_MODEL_NAME"
echo "Run dir:  $SFT_RUN_DIR"
echo "Slice:    $SLICE  cuda=$CUDA_TAG  cpus=$CPU_TAG"
echo "Started:  $(date)"
echo "==============="

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

/home/abhor/miniconda3/envs/agentic/bin/python scripts/training/sft/sft_train.py

echo "=== Finished at $(date) ==="
