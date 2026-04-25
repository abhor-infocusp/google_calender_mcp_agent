#!/usr/bin/env bash
# Dispatcher: run all-checkpoint eval on held-out test_data/ across 3 MIG slices in parallel.
#
# Layout:
#   Slice 1 (MIG-abbb3894) port 8006 -> SFT v6 (5 ckpts) — longest, merged dirs already exist
#   Slice 2 (MIG-dd607cdf) port 8007 -> DPO-from-SFT (3 ckpts, needs merge with SFT-merged base)
#   Slice 3 (MIG-7488039b) port 8008 -> DPO-from-Instruct (3 ckpts, needs merge with Qwen3-14B base)
#
# Qwen3-14B-Instruct baseline (no LoRA) runs AFTER DPO-from-Instruct finishes on slice 3.
#
# All evals run on test_data/ (50 held-out calendars, never seen by any model).
# Results go to $RUN_DIR/eval_test/{checkpoint-*.json,summary.csv}.

set -euo pipefail

PROJ=/home/abhor/google_calender_mcp_agent
cd "$PROJ"

PYTHON=/home/abhor/miniconda3/envs/agentic/bin/python

SLICE_1=MIG-abbb3894-4f8c-5e33-b602-6a485436950d
SLICE_2=MIG-dd607cdf-e8cb-531f-b478-417160625a35
SLICE_3=MIG-7488039b-0c78-50bb-8112-a1ae051fc3f7

# Run dirs
SFT_RUN=runs/sft_v6_qwen3_14b_20260420
DPO_SFT_RUN=runs/dpo_qwen3_14b_sft_20260423
DPO_INS_RUN=runs/dpo_qwen3_14b_instruct_20260423

# Base models (for merge)
BASE_QWEN=Qwen/Qwen3-14B
BASE_SFT_MERGED=$PROJ/runs/sft_v6_qwen3_14b_20260420/eval/merged_tmp_6212

# Ensure eval_test dirs exist
mkdir -p $SFT_RUN/eval_test/logs
mkdir -p $DPO_SFT_RUN/eval_test/logs
mkdir -p $DPO_INS_RUN/eval_test/logs

# For SFT v6, reuse the already-merged checkpoints via symlink (saves ~50 min of merging).
# Only symlink if target doesn't already exist.
for step in 1553 3106 4659 6212 7765; do
    src=$PROJ/$SFT_RUN/eval/merged_tmp_$step
    dst=$PROJ/$SFT_RUN/eval_test/merged_tmp_$step
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
        echo "Symlinked $dst -> $src"
    fi
done

echo "=== Launching 3 parallel orchestrators ==="

# ── Slice 1: SFT v6 (5 ckpts on test_data) ──
(
    export RUN_DIR=$SFT_RUN
    export EVAL_SUBDIR=eval_test
    export EVAL_MODE=test
    export NUM_CALENDARS=50
    export BASE_MODEL=$BASE_QWEN
    export MIG_UUID=$SLICE_1
    export EVAL_PORT=8006
    export SERVED_NAME=sft-v6-test-eval
    export PYTHONPATH=src
    export PYTHONUNBUFFERED=1
    cd $PROJ
    $PYTHON scripts/eval/eval_all_checkpoints.py \
        > $SFT_RUN/eval_test/logs/orchestrator.log 2>&1
) &
SLICE1_PID=$!
echo "Slice 1 (SFT v6): PID $SLICE1_PID"

# ── Slice 2: DPO-from-SFT (3 ckpts on test_data) ──
(
    export RUN_DIR=$DPO_SFT_RUN
    export EVAL_SUBDIR=eval_test
    export EVAL_MODE=test
    export NUM_CALENDARS=50
    export BASE_MODEL=$BASE_SFT_MERGED
    export MIG_UUID=$SLICE_2
    export EVAL_PORT=8007
    export SERVED_NAME=dpo-sft-test-eval
    export PYTHONPATH=src
    export PYTHONUNBUFFERED=1
    cd $PROJ
    $PYTHON scripts/eval/eval_all_checkpoints.py \
        > $DPO_SFT_RUN/eval_test/logs/orchestrator.log 2>&1
) &
SLICE2_PID=$!
echo "Slice 2 (DPO-from-SFT): PID $SLICE2_PID"

# ── Slice 3: DPO-from-Instruct (3 ckpts on test_data), THEN Instruct baseline ──
BASELINE_RUN=runs/qwen3_14b_instruct_baseline_20260424
mkdir -p $BASELINE_RUN/eval_test/logs
(
    export RUN_DIR=$DPO_INS_RUN
    export EVAL_SUBDIR=eval_test
    export EVAL_MODE=test
    export NUM_CALENDARS=50
    export BASE_MODEL=$BASE_QWEN
    export MIG_UUID=$SLICE_3
    export EVAL_PORT=8008
    export SERVED_NAME=dpo-ins-test-eval
    export PYTHONPATH=src
    export PYTHONUNBUFFERED=1
    cd $PROJ
    $PYTHON scripts/eval/eval_all_checkpoints.py \
        > $DPO_INS_RUN/eval_test/logs/orchestrator.log 2>&1

    # Follow-up: Qwen3-14B-Instruct baseline (no LoRA, just serve base directly).
    # Uses the same slice 3 now that DPO-from-Instruct evals are done.
    echo "[$(date)] Starting Instruct baseline eval" >> $BASELINE_RUN/eval_test/logs/baseline.log
    # Start vLLM serving Qwen3-14B directly — no merge, no LoRA.
    CUDA_VISIBLE_DEVICES=$SLICE_3 \
    $PYTHON -c "
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '$SLICE_3'
sys.argv = ['vllm', '--model', 'Qwen/Qwen3-14B', '--served-model-name', 'qwen3-14b-baseline',
            '--enable-auto-tool-choice', '--tool-call-parser', 'hermes',
            '--max-model-len', '4096', '--gpu-memory-utilization', '0.90',
            '--enforce-eager', '--quantization', 'fp8', '--port', '8008']
import runpy; runpy.run_module('vllm.entrypoints.openai.api_server', run_name='__main__')
" > $BASELINE_RUN/eval_test/logs/vllm.log 2>&1 &
    BASELINE_VLLM_PID=$!
    # Wait for server up (same pattern as eval_all_checkpoints)
    for i in $(seq 1 180); do
        if curl -sf http://localhost:8008/v1/models >/dev/null 2>&1; then
            echo "[$(date)] Baseline vLLM ready" >> $BASELINE_RUN/eval_test/logs/baseline.log
            break
        fi
        sleep 3
    done
    PYTHONPATH=src $PYTHON scripts/eval/eval_batch.py \
        --mode test --num-calendars 50 --max-queries 0 \
        --model qwen3-14b-baseline --base-url http://localhost:8008/v1 \
        --save $BASELINE_RUN/eval_test/baseline.json \
        >> $BASELINE_RUN/eval_test/logs/baseline.log 2>&1
    kill $BASELINE_VLLM_PID 2>/dev/null || true
) &
SLICE3_PID=$!
echo "Slice 3 (DPO-from-Instruct + baseline): PID $SLICE3_PID"

echo "$SLICE1_PID $SLICE2_PID $SLICE3_PID" > $PROJ/runs/logs/test_eval_pids.txt

echo ""
echo "Orchestrators launched. Monitor via:"
echo "  tail -F $SFT_RUN/eval_test/logs/orchestrator.log"
echo "  tail -F $DPO_SFT_RUN/eval_test/logs/orchestrator.log"
echo "  tail -F $DPO_INS_RUN/eval_test/logs/orchestrator.log"
echo ""
echo "PIDs (slice 1, 2, 3): $SLICE1_PID $SLICE2_PID $SLICE3_PID"
