#!/bin/bash
#SBATCH --job-name=judge-svc
#SBATCH --partition=main
#SBATCH --nodelist=azkaban
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
#SBATCH --time=72:00:00
#SBATCH --output=/home/abhor/google_calender_mcp_agent/runs/judge_service_20260501/logs/svc_%j.log
#SBATCH --error=/home/abhor/google_calender_mcp_agent/runs/judge_service_20260501/logs/svc_%j.log

# Long-lived judge service: vLLM (Qwen3-14B fp8 + hermes parser) on :$VLLM_PORT
# fronted by a FastAPI router that builds the prompt and returns just a verdict.
#
# RL training POSTs to http://127.0.0.1:$JUDGE_PORT/verdict.
#
# Env knobs:
#   SLICE=0          MIG slice index from slice_map.sh (default 0)
#   VLLM_PORT=8000   upstream vLLM port (default 8000)
#   JUDGE_PORT=8765  FastAPI port that RL talks to (default 8765)
#   MODEL_PATH=Qwen/Qwen3-14B  served weights (default Qwen3-14B base)

set -u
cd /home/abhor/google_calender_mcp_agent

SLICE="${SLICE:-0}"
VLLM_PORT="${VLLM_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8765}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"

RUN_DIR=/home/abhor/google_calender_mcp_agent/runs/judge_service_20260501
mkdir -p "$RUN_DIR/logs"

# ── Slice mapping ──────────────────────────────────────────────────
source scripts/training/common/slice_map.sh
export CUDA_VISIBLE_DEVICES="$(slice_cuda_uuid "$SLICE")"
TASKSET_CPUS="$(slice_cpu_range "$SLICE")"

PY=/home/abhor/miniconda3/envs/agentic/bin/python
VLLM=/home/abhor/miniconda3/envs/agentic/bin/vllm

# ── Thread caps + paths ────────────────────────────────────────────
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

# ── Per-run config that the FastAPI app reads ──────────────────────
export VLLM_BASE="http://127.0.0.1:${VLLM_PORT}/v1"
export VLLM_MODEL="judge"
export JUDGE_PORT
export JUDGE_LOG_DIR="$RUN_DIR"

# ── Metadata stamp (mirrors what run_telemetry.init_telemetry writes) ──
SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
cat > "$RUN_DIR/metadata.jsonl" <<EOF
{"event":"start","ts":"$(date -Iseconds)","script":"scripts/serving/judge_service.sbatch","slurm_job":"${SLURM_JOB_ID:-none}","slice":"$SLICE","cuda_uuid":"$CUDA_VISIBLE_DEVICES","cpus":"$TASKSET_CPUS","model":"$MODEL_PATH","vllm_port":$VLLM_PORT,"judge_port":$JUDGE_PORT,"git_sha":"$SHA","pid":$$}
EOF

cat <<EOF
[judge-svc] slice=$SLICE cuda=$CUDA_VISIBLE_DEVICES cpus=$TASKSET_CPUS
[judge-svc] vllm :$VLLM_PORT  fastapi :$JUDGE_PORT
[judge-svc] model=$MODEL_PATH log=$RUN_DIR
EOF

# ── Cleanup on exit ────────────────────────────────────────────────
VLLM_PID=""
SIDECAR_PID=""
cleanup() {
    echo "[judge-svc] terminating children…"
    [ -n "$SIDECAR_PID" ] && kill -TERM -- "-$SIDECAR_PID" 2>/dev/null
    [ -n "$VLLM_PID" ]    && kill -TERM -- "-$VLLM_PID"    2>/dev/null
    sleep 5
    [ -n "$SIDECAR_PID" ] && kill -KILL -- "-$SIDECAR_PID" 2>/dev/null
    [ -n "$VLLM_PID" ]    && kill -KILL -- "-$VLLM_PID"    2>/dev/null
    echo '{"event":"stop","ts":"'$(date -Iseconds)'"}' >> "$RUN_DIR/metadata.jsonl"
}
trap cleanup EXIT INT TERM

# ── Launch vLLM ────────────────────────────────────────────────────
VLLM_LOG="$RUN_DIR/logs/vllm.log"
echo "[judge-svc] starting vLLM → $VLLM_LOG"
setsid taskset -c "$TASKSET_CPUS" "$VLLM" serve "$MODEL_PATH" \
    --served-model-name "$VLLM_MODEL" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --quantization fp8 \
    --port "$VLLM_PORT" \
    >> "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

# ── Wait for vLLM to be ready ──────────────────────────────────────
echo "[judge-svc] waiting for vLLM /v1/models …"
for i in $(seq 1 600); do
    if curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
        echo "[judge-svc] vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[judge-svc] vLLM died during startup — see $VLLM_LOG"
        exit 1
    fi
    sleep 1
done

# ── Launch FastAPI sidecar in foreground (Slurm sees this as the job) ──
SIDECAR_LOG="$RUN_DIR/logs/sidecar.log"
echo "[judge-svc] starting FastAPI sidecar → $SIDECAR_LOG"
setsid taskset -c "$TASKSET_CPUS" "$PY" -m calendar_agent.judge.server \
    >> "$SIDECAR_LOG" 2>&1 &
SIDECAR_PID=$!

# Block on whichever exits first.
wait -n "$VLLM_PID" "$SIDECAR_PID"
rc=$?
echo "[judge-svc] one of {vllm=$VLLM_PID, sidecar=$SIDECAR_PID} exited rc=$rc"
exit $rc
