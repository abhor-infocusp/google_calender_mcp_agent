#!/bin/bash
#SBATCH --job-name=eval-sft-test
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=18:00:00
# stdout/err redirected via --output / --error set on the command line per-job

# Required env vars (passed via sbatch --export):
#   RUN_DIR    e.g. runs/sft_qwen3_8b_20260502
#   BASE_MODEL e.g. unsloth/qwen3-8b-unsloth-bnb-4bit
#   EVAL_PORT  unique per concurrent job
#   SERVED_NAME e.g. sft8b-eval

set -u
cd /home/abhor/google_calender_mcp_agent

: "${RUN_DIR:?RUN_DIR must be set}"
: "${BASE_MODEL:?BASE_MODEL must be set}"
: "${EVAL_PORT:?EVAL_PORT must be set}"
: "${SERVED_NAME:?SERVED_NAME must be set}"

mkdir -p "$RUN_DIR/eval_test/logs"

# Slurm has cgroup-pinned us to one MIG slice. CUDA_VISIBLE_DEVICES is set to
# a literal MIG UUID. eval_all_checkpoints.py overrides CUDA_VISIBLE_DEVICES
# with whatever MIG_UUID we give it, so feed it back the actual allocated UUID.
ACTUAL_MIG="${CUDA_VISIBLE_DEVICES:-}"
echo "Slurm allocated CUDA_VISIBLE_DEVICES=$ACTUAL_MIG" >&2
if [[ -z "$ACTUAL_MIG" || "$ACTUAL_MIG" != MIG-* ]]; then
    echo "ERROR: expected a MIG UUID in CUDA_VISIBLE_DEVICES, got '$ACTUAL_MIG'" >&2
    exit 2
fi

export EVAL_MODE=test
export EVAL_SUBDIR=eval_test
export NUM_CALENDARS=49
export MIG_UUID="$ACTUAL_MIG"
export RUN_DIR BASE_MODEL EVAL_PORT SERVED_NAME

# Vertex AI ADC for Gemini judge — picked up by google-cloud-aiplatform.
# (gemini-2.0-flash-001 only — never pro; eval_batch.py hard-codes the model.)

PYTHONPATH=src PYTHONUNBUFFERED=1 \
    /home/abhor/miniconda3/envs/agentic/bin/python -u \
    scripts/eval/eval_all_checkpoints.py \
    > "$RUN_DIR/eval_test/logs/driver_${SLURM_JOB_ID:-local}.log" 2>&1

echo "DONE: $RUN_DIR  (exit=$?)" >&2
