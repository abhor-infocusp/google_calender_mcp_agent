#!/bin/bash
#SBATCH --job-name=dpo-sft
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/dpo_qwen3_14b_sft_20260423/logs/sbatch_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/dpo_qwen3_14b_sft_20260423/logs/sbatch_%j.log

set -euo pipefail

mkdir -p /home/abhor/google_calender_mcp_agent/runs/dpo_qwen3_14b_sft_20260423/logs

echo "=== SLURM JOB INFO ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "======================"

export CUDA_VISIBLE_DEVICES=0

cd /home/abhor/google_calender_mcp_agent
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Experiment 2: DPO from the calendar-SFT'd checkpoint (ckpt-6212 @ 82.5%).
export DPO_BASE_MODEL="/home/abhor/google_calender_mcp_agent/runs/sft_v6_qwen3_14b_20260420/eval/merged_tmp_6212"
export DPO_PAIRS="runs/dpo/pairs_from_14b_rl.jsonl"
export DPO_OUTPUT_DIR="runs/dpo_qwen3_14b_sft_20260423"
export DPO_RUN_NAME="dpo_from_sft"

/home/abhor/miniconda3/envs/agentic/bin/python scripts/training/dpo_train.py

echo "=== Job finished at $(date) ==="
