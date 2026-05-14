#!/bin/bash
#SBATCH --job-name=eval-localjudge
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

# Eval one model on test_data/ using the deployed local judge service.
#
# Required env (pass via --export=ALL,...):
#   MODEL_HF      base model (HF id or local path; merged-LoRA dir for SFT ckpts)
#   RUN_DIR       e.g. runs/sft_v6_qwen3_14b_20260420  — outputs go to $RUN_DIR/eval_test_localjudge/
#   OUT_NAME      output basename, e.g. checkpoint-4659 or baseline (no .json)
#   EVAL_PORT     unique per concurrent job (e.g. 8011/8012/8013)
#   SERVED_NAME   served-model-name for vLLM
#
# Optional env (set both for RL LoRA serving — NEVER merge an RL LoRA):
#   LORA_PATH     path to adapter dir (with adapter_config.json)
#   LORA_NAME     name to register the LoRA under (defaults to "$SERVED_NAME-lora")
#   MAX_LORA_RANK 16 by default
#
# Judge service must already be running and reachable at $JUDGE_URL
# (default http://127.0.0.1:8765).

set -u
cd /home/abhor/google_calender_mcp_agent

: "${MODEL_HF:?MODEL_HF must be set}"
: "${RUN_DIR:?RUN_DIR must be set}"
: "${OUT_NAME:?OUT_NAME must be set}"
: "${EVAL_PORT:?EVAL_PORT must be set}"
: "${SERVED_NAME:?SERVED_NAME must be set}"

LORA_PATH="${LORA_PATH:-}"
LORA_NAME="${LORA_NAME:-${SERVED_NAME}-lora}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
JUDGE_URL="${JUDGE_URL:-http://127.0.0.1:8765}"

OUT_DIR="$RUN_DIR/eval_test_localjudge"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# vLLM model name as the eval script sees it: when LoRA is loaded, the eval
# must request the LoRA's served name (not the base).
EVAL_MODEL="$SERVED_NAME"
if [ -n "$LORA_PATH" ]; then
    EVAL_MODEL="$LORA_NAME"
fi

ACTUAL_MIG="${CUDA_VISIBLE_DEVICES:-}"
echo "[$(date)] Slurm allocated CUDA_VISIBLE_DEVICES=$ACTUAL_MIG" >&2
if [[ -z "$ACTUAL_MIG" || "$ACTUAL_MIG" != MIG-* ]]; then
    echo "ERROR: expected MIG UUID, got '$ACTUAL_MIG'" >&2
    exit 2
fi

# Verify judge is reachable before we burn GPU time spinning up vLLM.
if ! curl -sf "$JUDGE_URL/health" >/dev/null 2>&1; then
    echo "ERROR: judge service not reachable at $JUDGE_URL/health" >&2
    exit 4
fi

PYTHON=/home/abhor/miniconda3/envs/agentic/bin/python
LOG="$LOG_DIR/${OUT_NAME}.log"

{
    echo "[$(date)] === eval_with_local_judge ==="
    echo "MODEL_HF=$MODEL_HF"
    echo "LORA_PATH=${LORA_PATH:-(none)}"
    echo "RUN_DIR=$RUN_DIR  OUT_NAME=$OUT_NAME"
    echo "EVAL_PORT=$EVAL_PORT  SERVED_NAME=$SERVED_NAME  EVAL_MODEL=$EVAL_MODEL"
    echo "JUDGE_URL=$JUDGE_URL"
    echo "MIG=$ACTUAL_MIG"
} >> "$LOG"

# Build vLLM args
VLLM_ARGS=( serve "$MODEL_HF"
    --served-model-name "$SERVED_NAME"
    --enable-auto-tool-choice
    --tool-call-parser hermes
    --max-model-len 4096
    --gpu-memory-utilization 0.85
    --enforce-eager
    --quantization fp8
    --port "$EVAL_PORT" )

if [ -n "$LORA_PATH" ]; then
    VLLM_ARGS+=( --enable-lora
        --max-lora-rank "$MAX_LORA_RANK"
        --lora-modules "$LORA_NAME=$LORA_PATH" )
fi

VLLM=/home/abhor/miniconda3/envs/agentic/bin/vllm

CUDA_VISIBLE_DEVICES="$ACTUAL_MIG" PYTHONUNBUFFERED=1 \
    "$VLLM" "${VLLM_ARGS[@]}" > "$LOG_DIR/${OUT_NAME}.vllm.log" 2>&1 &
VLLM_PID=$!

# Wait for vLLM up
for i in $(seq 1 240); do
    if curl -sf "http://localhost:$EVAL_PORT/v1/models" >/dev/null 2>&1; then
        echo "[$(date)] vLLM ready (PID $VLLM_PID)" >> "$LOG"
        break
    fi
    sleep 3
done

if ! curl -sf "http://localhost:$EVAL_PORT/v1/models" >/dev/null 2>&1; then
    echo "[$(date)] ERROR: vLLM never came up" >> "$LOG"
    kill $VLLM_PID 2>/dev/null || true
    exit 3
fi

# Run eval
JUDGE_URL="$JUDGE_URL" PYTHONPATH=src $PYTHON scripts/eval/eval_batch.py \
    --mode test --num-calendars 50 --max-queries 0 \
    --judge local \
    --model "$EVAL_MODEL" --base-url "http://localhost:$EVAL_PORT/v1" \
    --save "$OUT_DIR/${OUT_NAME}.json" \
    >> "$LOG" 2>&1
RC=$?

kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "[$(date)] DONE rc=$RC out=$OUT_DIR/${OUT_NAME}.json" >> "$LOG"
exit $RC
