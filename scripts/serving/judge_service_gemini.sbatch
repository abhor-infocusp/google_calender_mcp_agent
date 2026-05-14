#!/bin/bash
#SBATCH --job-name=judge-svc-gemini
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=72:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_service_gemini_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_service_gemini_%j.log

# Gemini-backed judge service. NO GPU REQUIRED — calls Vertex AI.
# Drop-in for the local-vLLM judge: same /verdict and /health API on the
# same port (default 8765). RL training can swap by pointing JUDGE_URL at
# this host.
#
# Env knobs:
#   JUDGE_ROUTER     structured | router | qwen_v2 | gemini_v2  (default structured)
#   JUDGE_PORT       FastAPI port that RL talks to (default 8765)
#   GEMINI_MODEL     vertex model id (default gemini-2.0-flash-001)
#   JUDGE_LOG_DIR    where calls.jsonl goes
#
# Launch:
#   sbatch --export=ALL,JUDGE_ROUTER=structured \
#       scripts/serving/judge_service_gemini.sbatch

set -u
cd /home/abhor/google_calender_mcp_agent

JUDGE_PORT="${JUDGE_PORT:-8765}"
JUDGE_ROUTER="${JUDGE_ROUTER:-structured}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.0-flash-001}"

RUN_DIR=runs/judge_service_gemini_$(date +%Y%m%d)
mkdir -p "$RUN_DIR"

echo "=== Gemini judge service ==="
echo "Job:    ${SLURM_JOB_ID:-bare}"
echo "Port:   $JUDGE_PORT"
echo "Router: $JUDGE_ROUTER"
echo "Model:  $GEMINI_MODEL"
echo "Log:    $RUN_DIR/calls.jsonl"
echo "Start:  $(date)"
echo "============================"

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export JUDGE_PORT JUDGE_ROUTER GEMINI_MODEL
export JUDGE_LOG_DIR="$RUN_DIR"

exec /home/abhor/miniconda3/envs/agentic/bin/python -m calendar_agent.judge.server_gemini
