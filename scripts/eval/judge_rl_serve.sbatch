#!/bin/bash
#SBATCH --job-name=judge-rl-serve
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_filter_validation_20260505/logs/rl_serve_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_filter_validation_20260505/logs/rl_serve_%j.log

# Serve an RL LoRA adapter as a judge.
# Required env vars:
#   PORT          listen port (e.g. 8011, 8012)
#   BASE          base model path (HF id or local path)
#   LORA_NAME     served-model-name for the LoRA
#   LORA_PATH     path to the LoRA dir (with adapter_config.json)

set -u
cd /home/abhor/google_calender_mcp_agent

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    "$BASE" \
    --enable-lora \
    --max-lora-rank 16 \
    --lora-modules "$LORA_NAME=$LORA_PATH" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --quantization fp8 \
    --port "$PORT"
