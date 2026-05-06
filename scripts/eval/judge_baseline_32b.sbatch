#!/bin/bash
#SBATCH --job-name=judge-baseline-32b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_baseline_20260430/logs/sbatch_32b_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_baseline_20260430/logs/sbatch_32b_%j.log

set -u
cd /home/abhor/google_calender_mcp_agent

RUN_DIR=runs/judge_baseline_20260430
mkdir -p "$RUN_DIR/eval" "$RUN_DIR/logs"

# Qwen3-32B in bf16 ~64 GiB, won't fit a 24 GiB MIG slice. Use 4-bit nf4 (~18 GiB).
PYTHONPATH=src PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/python \
    scripts/eval/eval_judge_on_art.py \
    --checkpoint "" \
    --base-model Qwen/Qwen3-32B \
    --load-in-4bit \
    --num-samples 286 \
    --output "$RUN_DIR/eval/art_holdout_qwen3_32b_base.json"
