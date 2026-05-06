#!/bin/bash
#SBATCH --job-name=judge-sft14b-serve
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_sft14b_eval_20260504/logs/serve_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_sft14b_eval_20260504/logs/serve_%j.log

# Serve the SFT'd calendar agent (Qwen3-14B + LoRA, merged) AS a judge.
# Use the same router-qwen-v2 + /no_think prompt the deployed judge uses.
# Goal: see if SFT-on-tool-trajectories gives an independent 3rd opinion vs base-14B.
set -u
cd /home/abhor/google_calender_mcp_agent

PORT=8012
CKPT=/home/abhor/google_calender_mcp_agent/runs/sft_v6_qwen3_14b_20260420/eval/merged_tmp_4659

PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/vllm serve \
    "$CKPT" \
    --served-model-name sft14b-judge \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --quantization fp8 \
    --port "$PORT"
