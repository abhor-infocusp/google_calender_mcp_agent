#!/bin/bash
#SBATCH --job-name=rl-grpo-base
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/rl_grpo_qwen3_14b_base_20260426/logs/sbatch_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/rl_grpo_qwen3_14b_base_20260426/logs/sbatch_%j.log

set -u
cd /home/abhor/google_calender_mcp_agent

# vLLM 0.19 can't parse MIG UUIDs (slurm sets CUDA_VISIBLE_DEVICES=MIG-<uuid>).
# Override to integer index — slurm's cgroup still enforces GPU isolation.
export CUDA_VISIBLE_DEVICES=0

# Slurm's cgroup already pins us to --cpus-per-task cores, so taskset is
# redundant. Leaving TASKSET_CPUS unset disables auto_restart's pinning block.

# rl_train.py knobs
export RL_BASE_MODEL="Qwen/Qwen3-14B"
export RL_PROJECT="grpo-base-20260426"
export RL_MODEL_NAME="qwen3-14b-base"
export RL_VLLM_PORT="8005"

# auto_restart.sh knobs
export SCRIPT_PATH="scripts/training/rl/rl_train.py"
export RUN_DIR="runs/rl_grpo_qwen3_14b_base_20260426"
export MAX_HOURS=0
export MAX_RESTARTS=9999
export ART_DEADLOCK_TIMEOUT_S=600
export ART_DEADLOCK_HARD_CEILING_S=1800
export CHECKPOINT_KEEP_EVERY=500

mkdir -p "${RUN_DIR}/logs"
exec scripts/training/common/auto_restart.sh
