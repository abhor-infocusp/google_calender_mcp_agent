#!/bin/bash
#SBATCH --job-name=rl-orpo
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/rl_orpo_qwen3_14b_default/logs/sbatch_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/rl_orpo_qwen3_14b_default/logs/sbatch_%j.log

# ORPO trainer for the calendar agent — see docs/orpo/design.md.
# Single-MIG-slice: vLLM and training share the same GPU via ART's
# in-process sleep/wake plumbing. Run multiple instances in parallel on
# different MIG slices for ablations.

set -u
cd /home/abhor/google_calender_mcp_agent

# vLLM 0.19 can't parse MIG UUIDs in slurm-set CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES=0

# rl_orpo.py knobs
# Default: SFT v6 ckpt-4659 *merged into base* — see comment below the
# ART_INJECT_LORA_CHECKPOINT block. Pass plain "Qwen/Qwen3-14B" for a
# from-base ablation.
export RL_BASE_MODEL="${RL_BASE_MODEL:-runs/sft_v6_qwen3_14b_20260420/eval_test/merged_tmp_4659}"
export RL_PROJECT="${RL_PROJECT:-orpo-20260501}"
export RL_MODEL_NAME="${RL_MODEL_NAME:-qwen3-14b-orpo}"
export RL_VLLM_PORT="${RL_VLLM_PORT:-8009}"

# ORPO sampling + rollout hyperparameters (paper / design doc defaults).
export ORPO_N_QUERIES="${ORPO_N_QUERIES:-20}"
export ORPO_K_HARD="${ORPO_K_HARD:-8}"
export ORPO_K_EASY="${ORPO_K_EASY:-4}"
export ORPO_BUFFER_PER_SCENARIO="${ORPO_BUFFER_PER_SCENARIO:-4}"

# ORPO loss + optimizer
export ORPO_BETA="${ORPO_BETA:-0.1}"
export ORPO_LAMBDA="${ORPO_LAMBDA:-1.0}"
export ORPO_LR="${ORPO_LR:-5e-6}"
# Lowered iteratively after 2026-05-01 OOMs:
#   - batch=4: OOM at log_softmax(.float()), 19.9 GiB ask (fixed: chunked logsumexp)
#   - batch=2: OOM at chunked logsumexp, 594 MiB short of 23.62 GiB slice cap.
#     Process at 23.13 GiB (peft_model on GPU + concat forward (4, 4096, 152064)
#     bf16 logits ≈ 5 GiB + activations + grads + optimizer).
#   - batch=1: concat is (2, ...) → halves forward memory. grad_accum=16 to keep
#     effective batch ≈16.
export ORPO_PER_DEVICE_BATCH="${ORPO_PER_DEVICE_BATCH:-1}"
export ORPO_GRAD_ACCUM="${ORPO_GRAD_ACCUM:-16}"

# NOTE on initial weights:
# SFT v6 was trained at LoRA rank 64; ORPO uses rank 8. Loading rank-64
# adapter into rank-8 PEFT is impossible (state-dict shape mismatch — see
# /tmp/orpo_run_dir.txt incident, 2026-05-01). The fix per CLAUDE.md and
# rl_train.py: merge the SFT LoRA into the base, then point RL_BASE_MODEL
# at the merged fp16 model. We use the existing eval-time merge of
# ckpt-4659 above. No ART_INJECT_LORA_CHECKPOINT — fresh rank-8 LoRA on
# top of the SFT-merged base.

# auto_restart.sh knobs
export SCRIPT_PATH="scripts/training/rl/rl_orpo.py"
export RUN_DIR="${RUN_DIR:-runs/rl_orpo_qwen3_14b_default}"
export MAX_HOURS=0
export MAX_RESTARTS=9999
export ART_DEADLOCK_TIMEOUT_S=600
export ART_DEADLOCK_HARD_CEILING_S=1800
export CHECKPOINT_KEEP_EVERY=25
export CHECKPOINT_KEEP_LATEST=2

mkdir -p "${RUN_DIR}/logs"
exec scripts/training/common/auto_restart.sh
