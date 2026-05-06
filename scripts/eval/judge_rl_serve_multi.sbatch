#!/bin/bash
#SBATCH --job-name=judge-rl-multi
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_filter_validation_20260505/logs/rl_multi_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_filter_validation_20260505/logs/rl_multi_%j.log

# Serve a base model with MANY LoRA adapters preloaded.
# Required env:
#   PORT
#   BASE
#   LORA_MODULES   space-separated list of "name=path" pairs

set -u
cd /home/abhor/google_calender_mcp_agent

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    "$BASE" \
    --enable-lora \
    --max-lora-rank 16 \
    --max-loras 4 \
    --lora-modules $LORA_MODULES \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --quantization fp8 \
    --port "$PORT"
