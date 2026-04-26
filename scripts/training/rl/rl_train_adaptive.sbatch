#!/bin/bash
#SBATCH --job-name=rl-adapt-qwen3-14b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/rl_adaptive_qwen3_14b_20260424/logs/rl_train_adaptive_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/rl_adaptive_qwen3_14b_20260424/logs/rl_train_adaptive_%j.log

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

# Separate output directory so this run does not collide with the existing
# vanilla GRPO run in runs/rl_qwen3_14b_20260420/.
export RL_RUN_DIR="runs/rl_adaptive_qwen3_14b_20260424"
mkdir -p "${RL_RUN_DIR}/logs/debug"

/home/abhor/miniconda3/envs/agentic/bin/python scripts/training/rl/rl_train_adaptive.py

echo "=== Job finished at $(date) ==="
