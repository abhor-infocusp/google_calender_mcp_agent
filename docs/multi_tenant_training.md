# Multi-tenant RL training on `azkaban`

Multiple RL trainings can run concurrently on `azkaban`'s 4 MIG slices, but
**only if they're properly isolated**. Without isolation, a sibling training's
BLAS/OMP threads will oversubscribe CPU cores and starve our vLLM, causing
silent reward collapse (we hit this on 2026-04-25; ~600 steps lost).

This page is the operational protocol for launching trainings safely.

## Slice map

`scripts/training/slice_map.sh` is the single source of truth. Run it directly
to print the current mapping:

```
$ scripts/training/slice_map.sh
Host CPUs: 128, slices: 4, cores/slice: 32
  slice 0  cuda=MIG-5dc2f940-5003-58b0-a068-bede55f1d56f  cpus=0-31
  slice 1  cuda=MIG-abbb3894-4f8c-5e33-b602-6a485436950d  cpus=32-63
  slice 2  cuda=MIG-dd607cdf-e8cb-531f-b478-417160625a35  cpus=64-95
  slice 3  cuda=MIG-7488039b-0c78-50bb-8112-a1ae051fc3f7  cpus=96-127
```

If MIG is reconfigured, regenerate from `nvidia-smi -L` and edit
`slice_map.sh`'s `_MIG_UUIDS` array.

## Launching a training (the only correct way)

Always use `scripts/training/auto_restart.sh`. It encapsulates setsid, process-
group cleanup, deadlock retry (Patch G v2), thread caps, and CPU pinning.

```bash
source scripts/training/slice_map.sh
SLICE=0  # pick an unused slice

CUDA_VISIBLE_DEVICES=$(slice_cuda_uuid $SLICE) \
TASKSET_CPUS=$(slice_cpu_range $SLICE) \
SCRIPT_PATH=scripts/training/rl_train.py \
RUN_DIR=runs/rl_qwen3_14b_20260420 \
nohup scripts/training/auto_restart.sh \
    > runs/rl_qwen3_14b_20260420/logs/loop.log 2>&1 &
disown
```

The first three env vars are mandatory. Defaults for everything else are sane
(see `auto_restart.sh` for the full list, e.g. `MAX_HOURS`, `MAX_RESTARTS`,
`ART_DEADLOCK_TIMEOUT_S`, `CHECKPOINT_KEEP_EVERY`).

**Always set `TASKSET_CPUS`** matching the slice. Without it, threads from
this run can wander onto cores another training is using.

## Checking GPU + CPU before/after launch

```bash
# Which slices are occupied?
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv

# Which cores is a given pid pinned to?
taskset -pc <pid>

# Sibling list at our launch is recorded in metadata.jsonl
tail -1 runs/<run>/metadata.jsonl | jq .nvidia_smi_compute_apps
```

## Recovery after interference

If a training run is corrupted (reward collapse, bad gradients, etc.):

1. **Stop the run** cleanly: `kill -TERM <wrapper_pid>` then `kill -- -<py_pid>`
   to kill the process group. `auto_restart.sh` exits cleanly.
2. **Find a known-good checkpoint**:
   ```bash
   ls .art/<project>/models/<name>/checkpoints/
   # e.g. 0500/  1000/  1500/  ...  8500/  best/
   ```
   `Patch K` (since 2026-04-26) keeps every 500th checkpoint plus the
   best-by-reward one between milestones.
3. **Roll back**: ART resumes from the highest-numbered checkpoint, so to
   revert, delete the bad ones:
   ```bash
   # Example: roll back from 8500 (corrupt) to 8000 (good)
   rm -rf .art/calendar-agent/models/calendar-agent-001/checkpoints/8500
   ```
   Re-launch via `auto_restart.sh` — it'll resume from the now-newest checkpoint.

If the latest *milestone* is the bad one (rare — checkpoints are 500 steps
apart), pick the next-older milestone or the `best/` symlink.

## Validation: dual-launch test

The isolation works if two concurrent trainings on different slices don't slow
each other down measurably. To verify:

```bash
# 1. Solo: launch real-RL on slice 0, watch first 10 STEP SUMMARYs.
source scripts/training/slice_map.sh
CUDA_VISIBLE_DEVICES=$(slice_cuda_uuid 0) \
TASKSET_CPUS=$(slice_cpu_range 0) \
SCRIPT_PATH=scripts/training/rl_train.py \
RUN_DIR=runs/rl_qwen3_14b_20260420 \
nohup scripts/training/auto_restart.sh > runs/rl_qwen3_14b_20260420/logs/loop.log 2>&1 &
# Record mean tps and step wall-time for steps N..N+10.

# 2. Add a second tenant on slice 1.
mkdir -p runs/rl_small_qwen25_05b/logs/debug
CUDA_VISIBLE_DEVICES=$(slice_cuda_uuid 1) \
TASKSET_CPUS=$(slice_cpu_range 1) \
SCRIPT_PATH=scripts/training/rl_train_small.py \
RUN_DIR=runs/rl_small_qwen25_05b \
MAX_HOURS=1 \
nohup scripts/training/auto_restart.sh > runs/rl_small_qwen25_05b/logs/loop.log 2>&1 &

# 3. Watch real-RL's next 10 STEP SUMMARYs. tps and step wall-time should
#    stay within ~10% of the solo baseline. (Without isolation: 50% drop.)
```

## Anti-patterns (what NOT to do)

- ❌ `python scripts/training/rl_train.py` directly without setsid. A deadlock
  leaves orphaned vLLM EngineCore processes consuming GPU memory.
- ❌ Set `CUDA_VISIBLE_DEVICES=0,1,2,3` or all-slices. Pick exactly one.
- ❌ Skip `TASKSET_CPUS` "just for one quick run". Threads spread across all
  128 cores and starve any other training that's running.
- ❌ `OMP_NUM_THREADS` unset. PyTorch defaults to `nproc` (128 on azkaban).
- ❌ Edit `rl_train.py` to call `delete_checkpoints()` unconditionally. Patch K
  is what saved us from the 2026-04-25 incident; don't undo it.
