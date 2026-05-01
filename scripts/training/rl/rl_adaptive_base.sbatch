#!/bin/bash
#SBATCH --job-name=rl-adaptive-base
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/rl_adaptive_qwen3_14b_base_20260426/logs/sbatch_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/rl_adaptive_qwen3_14b_base_20260426/logs/sbatch_%j.log

set -u
cd /home/abhor/google_calender_mcp_agent

# vLLM 0.19 MIG-UUID parse bug — see rl_grpo_base.sbatch.
export CUDA_VISIBLE_DEVICES=0

export RL_BASE_MODEL="Qwen/Qwen3-14B"
export RL_PROJECT="adaptive-base-20260426"
export RL_MODEL_NAME="qwen3-14b-adaptive"
export RL_VLLM_PORT="8007"

export SCRIPT_PATH="scripts/training/rl/rl_train_adaptive.py"
export RUN_DIR="runs/rl_adaptive_qwen3_14b_base_20260426"
export MAX_HOURS=0
export MAX_RESTARTS=9999
export ART_DEADLOCK_TIMEOUT_S=600
export ART_DEADLOCK_HARD_CEILING_S=1800
export CHECKPOINT_KEEP_EVERY=500

mkdir -p "${RUN_DIR}/logs"
exec scripts/training/common/auto_restart.sh
