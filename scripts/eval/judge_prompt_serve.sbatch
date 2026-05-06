#!/bin/bash
#SBATCH --job-name=judge-prompt-serve
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_%j.log

# Serve Qwen3-14B base via vLLM for prompt-tuning iteration.
# Port 8011 chosen to avoid colliding with anything else on azkaban.
#
# Once "Application startup complete" lands in the log, query at
#   http://azkaban:8011/v1/chat/completions
# with model="judge".

set -u
cd /home/abhor/google_calender_mcp_agent

PORT=8011

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    Qwen/Qwen3-14B \
    --served-model-name judge \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --quantization fp8 \
    --port "$PORT"
