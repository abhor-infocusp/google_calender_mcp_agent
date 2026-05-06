#!/bin/bash
#SBATCH --job-name=judge-cudagraphs
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=06:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_cudagraphs_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_cudagraphs_%j.log

# 14B fp8 with cudagraphs enabled (no --enforce-eager) on port 8015.
set -u
cd /home/abhor/google_calender_mcp_agent

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    Qwen/Qwen3-14B \
    --served-model-name judge \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --quantization fp8 \
    --port 8015
