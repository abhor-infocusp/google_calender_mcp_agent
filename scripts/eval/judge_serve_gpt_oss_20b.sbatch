#!/bin/bash
#SBATCH --job-name=judge-gptoss-svc
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_gptoss_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_prompt_tune_20260430/logs/serve_gptoss_%j.log

# Serve openai/gpt-oss-20b via vLLM on port 8020 for judge benchmarking.
# MXFP4 weights ship in-repo; vLLM auto-detects, no --quantization flag.
# Slice 1 — slice 0 is the long-lived production judge-svc.

set -u
cd /home/abhor/google_calender_mcp_agent

source scripts/training/common/slice_map.sh
SLICE="${SLICE:-1}"
export CUDA_VISIBLE_DEVICES="$(slice_cuda_uuid "$SLICE")"

PORT=8020
echo "[gptoss-svc] slice=$SLICE cuda=$CUDA_VISIBLE_DEVICES port=$PORT"

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    openai/gpt-oss-20b \
    --served-model-name judge-gptoss \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --max-num-seqs 32 \
    --port "$PORT"
