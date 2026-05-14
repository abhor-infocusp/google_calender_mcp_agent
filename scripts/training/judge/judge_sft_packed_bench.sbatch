#!/bin/bash
#SBATCH --job-name=judge-pack-bench
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --nodelist=azkaban

set -euo pipefail

: "${JUDGE_MODEL_NAME:?required}"
: "${JUDGE_RUN_DIR:?required}"
: "${JUDGE_TRAIN_JSONL:?required}"

cd /home/abhor/google_calender_mcp_agent
mkdir -p "$JUDGE_RUN_DIR/logs" "$JUDGE_RUN_DIR/diagnostics"
LOG="$JUDGE_RUN_DIR/logs/bench.log"
exec >>"$LOG" 2>&1

# Force a specific MIG slice (avoid contention with concurrent training).
if [ -n "${FORCE_MIG_UUID:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$FORCE_MIG_UUID"
fi

echo "=== JUDGE PACKING BENCH ==="
echo "Job:    ${SLURM_JOB_ID:-bare}"
echo "Cuda:   ${CUDA_VISIBLE_DEVICES:-slurm-managed}"
echo "--- isolation check (devices visible to this cgroup) ---"
nvidia-smi -L
echo "--- /proc/self/status DeviceCgroup ---"
grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/self/status || true
echo "--- torch view ---"
/home/abhor/miniconda3/envs/agentic/bin/python -c "import torch; print('torch.cuda.device_count():', torch.cuda.device_count()); print('torch.cuda.get_device_name(0):', torch.cuda.get_device_name(0) if torch.cuda.device_count() else 'NONE'); import os; print('UUID env:', os.environ.get('CUDA_VISIBLE_DEVICES'))" 2>&1 | head
echo "------------------------------------------------"
echo "Model:  $JUDGE_MODEL_NAME"
echo "RunDir: $JUDGE_RUN_DIR"
echo "Start:  $(date)"
echo "============================"

export PYTHONPATH=src
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

/home/abhor/miniconda3/envs/agentic/bin/python scripts/training/judge/judge_sft_train_packed.py

echo "=== Done $(date) ==="
