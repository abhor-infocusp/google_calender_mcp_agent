#!/bin/bash
#SBATCH --job-name=rl-qwen3-14b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/logs/rl_train_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/logs/rl_train_%j.log

set -euo pipefail

echo "=== SLURM JOB INFO ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURM_NODELIST"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo "======================"

# vLLM can't parse MIG UUIDs — Slurm sets CUDA_VISIBLE_DEVICES=MIG-<uuid>.
# Override to integer index so vLLM/torch see the device correctly.
# Slurm still enforces the cgroup GPU isolation regardless of this variable.
export CUDA_VISIBLE_DEVICES=0

cd /home/abhor/google_calender_mcp_agent

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

/home/abhor/miniconda3/envs/agentic/bin/python scripts/training/rl_train.py

echo "=== Job finished at $(date) ==="
