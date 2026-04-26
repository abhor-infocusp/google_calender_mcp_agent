#!/bin/bash
#SBATCH --job-name=dpo-instruct
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/dpo_qwen3_14b_instruct_20260423/logs/sbatch_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/dpo_qwen3_14b_instruct_20260423/logs/sbatch_%j.log

set -euo pipefail

mkdir -p /home/abhor/google_calender_mcp_agent/runs/dpo_qwen3_14b_instruct_20260423/logs

echo "=== SLURM JOB INFO ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "======================"

# Slurm binds a MIG slice; override to integer index (bnb/flash-attn need this).
export CUDA_VISIBLE_DEVICES=0

cd /home/abhor/google_calender_mcp_agent
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Experiment 1: DPO from the un-fine-tuned Qwen3-14B-Instruct.
export DPO_BASE_MODEL="Qwen/Qwen3-14B"
export DPO_PAIRS="runs/dpo/pairs_from_14b_rl.jsonl"
export DPO_OUTPUT_DIR="runs/dpo_qwen3_14b_instruct_20260423"
export DPO_RUN_NAME="dpo_from_instruct"

/home/abhor/miniconda3/envs/agentic/bin/python scripts/training/dpo/dpo_train.py

echo "=== Job finished at $(date) ==="
