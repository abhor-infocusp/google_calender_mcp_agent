#!/bin/bash
#SBATCH --job-name=judge-baseline-8b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_baseline_20260430/logs/sbatch_8b_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_baseline_20260430/logs/sbatch_8b_%j.log

set -u
cd /home/abhor/google_calender_mcp_agent

RUN_DIR=runs/judge_baseline_20260430
mkdir -p "$RUN_DIR/eval" "$RUN_DIR/logs"

PYTHONPATH=src PYTHONUNBUFFERED=1 /home/abhor/miniconda3/envs/agentic/bin/python \
    scripts/eval/eval_judge_on_art.py \
    --checkpoint "" \
    --base-model Qwen/Qwen3-8B \
    --num-samples 286 \
    --output "$RUN_DIR/eval/art_holdout_qwen3_8b_base.json"
