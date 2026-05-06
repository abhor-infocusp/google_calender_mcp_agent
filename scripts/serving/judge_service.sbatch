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
#   MODEL_PATH=Qwen/Qwen3-14B   base weights served by vLLM (HF id or local)
#   LORA_PATH=""                 if set, served with --enable-lora and the
#                                LoRA loaded as model name $VLLM_MODEL.
#                                NEVER merge an RL LoRA — always serve via LoRA.
#   MAX_LORA_RANK=16             passed to --max-lora-rank when LORA_PATH set

set -u
cd /home/abhor/google_calender_mcp_agent

SLICE="${SLICE:-0}"
VLLM_PORT="${VLLM_PORT:-8000}"
JUDGE_PORT="${JUDGE_PORT:-8765}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
# LORA_PATH is required: the deployed judge is rl-sft-4952 (LoRA on SFT v6
# merged base). No silent fallback to base-only — fail loudly if unset.
if [ -z "${LORA_PATH:-}" ]; then
    echo "[judge-svc] ERROR: LORA_PATH must be set. Refusing to launch a base-only judge." >&2
    exit 2
fi

RUN_DIR=/home/abhor/google_calender_mcp_agent/runs/judge_service_20260501
mkdir -p "$RUN_DIR/logs"

# ── Slice mapping ──────────────────────────────────────────────────
# Under Slurm, --gres=gpu:1 already pins us to one MIG slice via cgroup, and
# Slurm's cpuset restricts which CPUs we can run on. Overriding either one
# blindly causes failures when Slurm's allocation doesn't match SLICE
# (torch sees 0 devices; taskset rejects out-of-cgroup CPUs). Trust Slurm.
source scripts/training/common/slice_map.sh
if [ -n "${SLURM_JOB_ID:-}" ]; then
    # Slurm controls slice + cpuset. Don't override.
    CUDA_TAG="${CUDA_VISIBLE_DEVICES:-slurm-managed}"
    TASKSET_CPUS=""  # empty → skip taskset
else
    export CUDA_VISIBLE_DEVICES="$(slice_cuda_uuid "$SLICE")"
    CUDA_TAG="$CUDA_VISIBLE_DEVICES"
    TASKSET_CPUS="$(slice_cpu_range "$SLICE")"
fi

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
{"event":"start","ts":"$(date -Iseconds)","script":"scripts/serving/judge_service.sbatch","slurm_job":"${SLURM_JOB_ID:-none}","slice":"$SLICE","cuda_uuid":"$CUDA_TAG","cpus":"$TASKSET_CPUS","model":"$MODEL_PATH","vllm_port":$VLLM_PORT,"judge_port":$JUDGE_PORT,"git_sha":"$SHA","pid":$$}
EOF

cat <<EOF
[judge-svc] slice=$SLICE cuda=$CUDA_TAG cpus=${TASKSET_CPUS:-slurm-managed}
[judge-svc] vllm :$VLLM_PORT  fastapi :$JUDGE_PORT
[judge-svc] model=$MODEL_PATH log=$RUN_DIR
EOF

# ── Cleanup on exit ────────────────────────────────────────────────
VLLM_PID=""
SIDECAR_PID=""
WATCHDOG_PID=""
cleanup() {
    echo "[judge-svc] terminating children…"
    [ -n "$WATCHDOG_PID" ] && kill -TERM "$WATCHDOG_PID" 2>/dev/null
    [ -n "$SIDECAR_PID" ] && kill -TERM -- "-$SIDECAR_PID" 2>/dev/null
    [ -n "$VLLM_PID" ]    && kill -TERM -- "-$VLLM_PID"    2>/dev/null
    sleep 5
    [ -n "$SIDECAR_PID" ] && kill -KILL -- "-$SIDECAR_PID" 2>/dev/null
    [ -n "$VLLM_PID" ]    && kill -KILL -- "-$VLLM_PID"    2>/dev/null
    echo '{"event":"stop","ts":"'$(date -Iseconds)'"}' >> "$RUN_DIR/metadata.jsonl"
}
trap cleanup EXIT INT TERM

WATCHDOG_LOG="$RUN_DIR/logs/watchdog.jsonl"
: > "$WATCHDOG_LOG"

# ── Launch vLLM ────────────────────────────────────────────────────
VLLM_LOG="$RUN_DIR/logs/vllm.log"
echo "[judge-svc] starting vLLM → $VLLM_LOG"
if [ -n "$TASKSET_CPUS" ]; then VLLM_PIN=(taskset -c "$TASKSET_CPUS"); else VLLM_PIN=(); fi
echo "[judge-svc] serving base=$MODEL_PATH + LoRA=$LORA_PATH (name=$VLLM_MODEL, rank=$MAX_LORA_RANK)"
setsid "${VLLM_PIN[@]}" "$VLLM" serve "$MODEL_PATH" \
    --enable-lora --max-lora-rank "$MAX_LORA_RANK" \
    --lora-modules "${VLLM_MODEL}=${LORA_PATH}" \
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
setsid "${VLLM_PIN[@]}" "$PY" -m calendar_agent.judge.server \
    >> "$SIDECAR_LOG" 2>&1 &
SIDECAR_PID=$!

# ── Watchdog: per-10s GPU + host-mem + cgroup + Xid + process snapshot ──
# Started AFTER VLLM_PID and SIDECAR_PID are set — the subshell snapshots
# them at fork time, so launching earlier makes liveness checks see "".
# When vLLM EngineCore dies silently (no Python traceback in vllm.log) the
# breadcrumb usually shows up here:
#   - cgroup memory.current approaching memory.max → cgroup OOM-kill (SIGKILL)
#   - /dev/shm.used spiking → vLLM IPC backing store exhausted
#   - nvidia-smi Xid counter incrementing → driver-level GPU fault
#   - mem_avail_kb dropping → host-wide OOM pressure
# Cgroup path is read from /proc/self/cgroup once (cgroup v2 unified hierarchy).
CG_REL=$(awk -F: '/^0::/{print $3; exit}' /proc/self/cgroup 2>/dev/null)
CG_DIR="/sys/fs/cgroup${CG_REL}"
(
    while sleep 10; do
        ts=$(date -Iseconds)
        # nvidia-smi: memory.used MiB, free MiB, utilization %
        gpu=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        # ECC uncorrectable counters — proxy for some Xid events on this driver
        # (Xid is dmesg-only, needs sudo). Volatile=since-boot. Two ints "vol,agg".
        ecc_unc=$(nvidia-smi -q -d ECC 2>/dev/null | awk '
            /Volatile/{vol=1; agg=0} /Aggregate/{agg=1; vol=0}
            /SRAM Uncorrectable SEC-DED|DRAM Uncorrectable/{
                if(vol) v+=$NF; else if(agg) a+=$NF
            }
            END{printf "%d,%d", v+0, a+0}')
        # host mem
        mem_avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
        mem_total_kb=$(awk '/MemTotal/ {print $2}' /proc/meminfo)
        load=$(awk '{print $1","$2","$3}' /proc/loadavg)
        # cgroup mem (Slurm --mem=80G enforced here; OOM-kill victim picked from this cgroup)
        cg_cur=$(cat "$CG_DIR/memory.current" 2>/dev/null || echo 0)
        cg_max=$(cat "$CG_DIR/memory.max" 2>/dev/null || echo 0)
        cg_oom=$(awk '/^oom_kill /{print $2}' "$CG_DIR/memory.events" 2>/dev/null || echo 0)
        # /dev/shm — vLLM v1 IPC between APIServer and EngineCore lives here
        shm_used_kb=$(df -k /dev/shm 2>/dev/null | awk 'NR==2{print $3}')
        shm_avail_kb=$(df -k /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
        # PID liveness + RSS
        vllm_alive=0; vllm_rss=0
        if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
            vllm_alive=1
            vllm_rss=$(awk '/VmRSS/ {print $2}' "/proc/$VLLM_PID/status" 2>/dev/null || echo 0)
        fi
        sidecar_alive=0
        [ -n "$SIDECAR_PID" ] && kill -0 "$SIDECAR_PID" 2>/dev/null && sidecar_alive=1
        # EngineCore subprocess (vLLM v1 spawns it; find by parent=VLLM_PID)
        engine_pids=$(pgrep -P "$VLLM_PID" 2>/dev/null | tr '\n' ',' | sed 's/,$//')
        printf '{"ts":"%s","gpu_mem_used_free_util":"%s","ecc_unc_vol_agg":"%s","mem_avail_kb":%s,"mem_total_kb":%s,"loadavg":[%s],"cg_mem_current":%s,"cg_mem_max":"%s","cg_oom_kill":%s,"shm_used_kb":%s,"shm_avail_kb":%s,"vllm_alive":%d,"vllm_rss_kb":%s,"sidecar_alive":%d,"engine_pids":"%s"}\n' \
            "$ts" "$gpu" "$ecc_unc" "$mem_avail_kb" "$mem_total_kb" "$load" \
            "$cg_cur" "$cg_max" "$cg_oom" "$shm_used_kb" "$shm_avail_kb" \
            "$vllm_alive" "$vllm_rss" "$sidecar_alive" "$engine_pids" \
            >> "$WATCHDOG_LOG"
    done
) &
WATCHDOG_PID=$!
echo "[judge-svc] watchdog pid=$WATCHDOG_PID cgroup=$CG_DIR → $WATCHDOG_LOG"

# Block on whichever exits first.
wait -n "$VLLM_PID" "$SIDECAR_PID"
rc=$?
echo "[judge-svc] one of {vllm=$VLLM_PID, sidecar=$SIDECAR_PID} exited rc=$rc"

# ── Post-mortem dump (best-effort; runs only when something died) ──
PM="$RUN_DIR/logs/postmortem_${SLURM_JOB_ID:-bare}_$(date +%s).log"
{
    echo "=== POST-MORTEM $(date -Iseconds) — exit rc=$rc ==="
    echo "--- liveness ---"
    for name in VLLM SIDECAR; do
        pidvar="${name}_PID"; pid="${!pidvar}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "$name pid=$pid alive"; else echo "$name pid=$pid DEAD"; fi
    done
    echo "--- nvidia-smi ---"
    nvidia-smi 2>&1 | head -40 || true
    echo "--- nvidia-smi ECC (Xid is dmesg-only, may not be readable) ---"
    nvidia-smi -q -d ECC 2>&1 | grep -iE "uncorrect|aggregate|volatile" | head -30 || true
    journalctl -k --since "5 minutes ago" 2>&1 | grep -iE "xid|nvidia|nvrm" | head -20 || echo "(no journal access)"
    echo "--- /proc/meminfo (top) ---"
    head -8 /proc/meminfo
    echo "--- cgroup memory ($CG_DIR) ---"
    for f in memory.current memory.max memory.events memory.peak; do
        v=$(cat "$CG_DIR/$f" 2>/dev/null || echo "(n/a)")
        echo "  $f: $v"
    done
    echo "--- /dev/shm ---"
    df -h /dev/shm 2>&1 | head
    echo "--- loadavg ---"
    cat /proc/loadavg
    echo "--- last 30 watchdog lines ---"
    tail -30 "$WATCHDOG_LOG" 2>/dev/null
    echo "--- last 60 vllm.log lines ---"
    tail -60 "$VLLM_LOG" 2>/dev/null
    echo "--- dmesg tail (best-effort, may need sudo) ---"
    dmesg -T 2>&1 | tail -40 || echo "(no dmesg perms)"
} > "$PM" 2>&1
echo "[judge-svc] post-mortem → $PM"
exit $rc
