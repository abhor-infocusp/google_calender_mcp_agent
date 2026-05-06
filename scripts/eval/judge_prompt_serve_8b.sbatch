#!/bin/bash
#SBATCH --job-name=judge-bench-8b
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_8b_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_8b_%j.log

# Qwen3-8B fp8 server on port 8014 for latency benchmarks.
set -u
cd /home/abhor/google_calender_mcp_agent

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    Qwen/Qwen3-8B \
    --served-model-name judge-8b \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --quantization fp8 \
    --port 8014
