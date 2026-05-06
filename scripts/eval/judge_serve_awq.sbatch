#!/bin/bash
#SBATCH --job-name=judge-awq
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_awq_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_awq_%j.log

# Qwen3-14B AWQ (4-bit) on port 8017. No --enforce-eager (cudagraphs ON).
set -u
cd /home/abhor/google_calender_mcp_agent

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    Qwen/Qwen3-14B-AWQ \
    --served-model-name judge \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --quantization awq \
    --port 8017
