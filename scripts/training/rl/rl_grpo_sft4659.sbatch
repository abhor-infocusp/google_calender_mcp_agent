#!/bin/bash
#SBATCH --job-name=rl-grpo-sft4659
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/rl_grpo_qwen3_14b_sft4659_20260426/logs/sbatch_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/rl_grpo_qwen3_14b_sft4659_20260426/logs/sbatch_%j.log

set -u
cd /home/abhor/google_calender_mcp_agent

# vLLM 0.19 MIG-UUID parse bug — see rl_grpo_base.sbatch.
export CUDA_VISIBLE_DEVICES=0

# Base model = SFT v6 ckpt-4659 merged into Qwen3-14B.
# This dir is gitignored — verify it exists before submitting.
export RL_BASE_MODEL="/home/abhor/google_calender_mcp_agent/runs/sft_v6_qwen3_14b_20260420/eval/merged_tmp_4659"
export RL_PROJECT="grpo-sft4659-20260426"
export RL_MODEL_NAME="qwen3-14b-sft4659"
export RL_VLLM_PORT="8006"

export SCRIPT_PATH="scripts/training/rl/rl_train.py"
export RUN_DIR="runs/rl_grpo_qwen3_14b_sft4659_20260426"
export MAX_HOURS=0
export MAX_RESTARTS=9999
export ART_DEADLOCK_TIMEOUT_S=600
export ART_DEADLOCK_HARD_CEILING_S=1800
export CHECKPOINT_KEEP_EVERY=500

if [ ! -d "$RL_BASE_MODEL" ]; then
    echo "[sbatch] FATAL: RL_BASE_MODEL not found: $RL_BASE_MODEL" >&2
    echo "[sbatch] re-merge SFT ckpt-4659 with scripts/training/common/merge_lora.py" >&2
    exit 1
fi

mkdir -p "${RUN_DIR}/logs"
exec scripts/training/common/auto_restart.sh
