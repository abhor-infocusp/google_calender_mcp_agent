#!/bin/bash
#SBATCH --job-name=instruct-baseline
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=04:00:00

# Required env (passed via --export):
#   MODEL_HF      e.g. Qwen/Qwen3-8B
#   RUN_DIR       e.g. runs/qwen3_8b_instruct_baseline_20260505
#   EVAL_PORT     unique per concurrent job (e.g. 8011/8012/8013)
#   SERVED_NAME   e.g. qwen3-8b-baseline

set -u
cd /home/abhor/google_calender_mcp_agent

: "${MODEL_HF:?MODEL_HF must be set}"
: "${RUN_DIR:?RUN_DIR must be set}"
: "${EVAL_PORT:?EVAL_PORT must be set}"
: "${SERVED_NAME:?SERVED_NAME must be set}"

mkdir -p "$RUN_DIR/eval_test/logs"

ACTUAL_MIG="${CUDA_VISIBLE_DEVICES:-}"
echo "Slurm allocated CUDA_VISIBLE_DEVICES=$ACTUAL_MIG" >&2
if [[ -z "$ACTUAL_MIG" || "$ACTUAL_MIG" != MIG-* ]]; then
    echo "ERROR: expected MIG UUID, got '$ACTUAL_MIG'" >&2
    exit 2
fi

PYTHON=/home/abhor/miniconda3/envs/agentic/bin/python
LOG_DIR="$RUN_DIR/eval_test/logs"

echo "[$(date)] Starting Instruct baseline eval for $MODEL_HF" >> "$LOG_DIR/baseline.log"

# Serve base model directly (no LoRA, no merge), fp8 quant, hermes tool parser.
CUDA_VISIBLE_DEVICES="$ACTUAL_MIG" \
$PYTHON -c "
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '$ACTUAL_MIG'
sys.argv = ['vllm', '--model', '$MODEL_HF', '--served-model-name', '$SERVED_NAME',
            '--enable-auto-tool-choice', '--tool-call-parser', 'hermes',
            '--max-model-len', '4096', '--gpu-memory-utilization', '0.85',
            '--enforce-eager', '--quantization', 'fp8', '--port', '$EVAL_PORT']
import runpy; runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
" > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!

# Wait for server up
for i in $(seq 1 240); do
    if curl -sf "http://localhost:$EVAL_PORT/v1/models" >/dev/null 2>&1; then
        echo "[$(date)] Baseline vLLM ready (PID $VLLM_PID)" >> "$LOG_DIR/baseline.log"
        break
    fi
    sleep 3
done

if ! curl -sf "http://localhost:$EVAL_PORT/v1/models" >/dev/null 2>&1; then
    echo "[$(date)] vLLM never came up — aborting" >> "$LOG_DIR/baseline.log"
    kill $VLLM_PID 2>/dev/null || true
    exit 3
fi

PYTHONPATH=src $PYTHON scripts/eval/eval_batch.py \
    --mode test --num-calendars 50 --max-queries 0 \
    --model "$SERVED_NAME" --base-url "http://localhost:$EVAL_PORT/v1" \
    --save "$RUN_DIR/eval_test/baseline.json" \
    >> "$LOG_DIR/baseline.log" 2>&1
RC=$?

kill $VLLM_PID 2>/dev/null || true
echo "[$(date)] DONE rc=$RC for $MODEL_HF" >> "$LOG_DIR/baseline.log"
exit $RC
