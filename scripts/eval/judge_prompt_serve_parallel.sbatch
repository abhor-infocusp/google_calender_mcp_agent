#!/bin/bash
#SBATCH --job-name=judge-prompt-serve
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_%j.log

# Same as judge_prompt_serve.sbatch but takes PORT env var so we can run
# multiple in parallel on different MIG slices. Submit with:
#   PORT=8012 sbatch --export=ALL,PORT scripts/eval/judge_prompt_serve_parallel.sbatch

set -u
cd /home/abhor/google_calender_mcp_agent

PORT="${PORT:-8011}"

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    Qwen/Qwen3-14B \
    --served-model-name judge \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --quantization fp8 \
    --port "$PORT"
