#!/bin/bash
#SBATCH --job-name=judge-v3-eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nodelist=azkaban

set -euo pipefail
cd /home/abhor/google_calender_mcp_agent

RUN_DIR="${RUN_DIR:-runs/judge_v3_qwen3_14b_20260507}"
LORA="${LORA:-$RUN_DIR/checkpoints/final}"
OUT="${OUT:-$RUN_DIR/eval/manual_oracle.json}"

mkdir -p "$(dirname "$OUT")" "$RUN_DIR/logs"
LOG="$RUN_DIR/logs/eval_manual_oracle.log"
exec >>"$LOG" 2>&1

echo "=== JUDGE V3 MANUAL ORACLE EVAL ==="
echo "Job:    ${SLURM_JOB_ID:-bare}"
echo "Cuda:   ${CUDA_VISIBLE_DEVICES:-slurm-managed}"
echo "Lora:   $LORA"
echo "Out:    $OUT"
echo "Start:  $(date)"
echo "==================================="

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

/home/abhor/miniconda3/envs/agentic/bin/python scripts/eval/judge_v3_manual_eval.py \
    --base Qwen/Qwen3-14B \
    --lora "$LORA" \
    --out "$OUT"

echo "=== Done $(date) ==="
