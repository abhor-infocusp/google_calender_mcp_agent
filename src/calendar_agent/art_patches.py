"""Runtime monkey-patches for ART 0.5.17.

Apply these before importing ART to fix:
- Entropy in autograd graph wastes memory (patch D)
- chunk_size assertion fails on ragged sequences (patch D)
- Health monitor timeout destroys cached service → OOM (patch E)

Patches A, B, C from ART 0.5.4 are no longer needed:
- A (asyncio.Queue → queue.Queue): ART 0.5.17 uses nest_asyncio + asyncio.wait
- B (gc/empty_cache in train_mode): fixed upstream in _train_shared()
- C (run_in_executor + call_soon_threadsafe): ART 0.5.17 uses nest_asyncio

Usage:
    import calendar_agent.art_patches  # patches applied on import
    import art  # now safe to use
"""

import asyncio
import re
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

_APPLIED: set[str] = set()

# Set this before importing art to inject RL LoRA weights into the fresh adapter.
# Path should point to a directory with adapter_model.safetensors.
INJECT_LORA_CHECKPOINT: str | None = None


def _log(name: str) -> None:
    _APPLIED.add(name)
    print(f"[art_patches] Applied: {name}")


# ── Patch D: entropy detach + relax chunk_size assertion (train.py) ───


def _patch_calculate_logprobs():
    """Detach entropy from autograd graph and remove chunk_size assertion.

    Entropy is only used for logging metrics, not loss. Keeping it in the
    graph wastes memory. The seq_len % chunk_size == 0 assertion fails on
    ragged sequences; the loop already handles partial last chunks.
    """
    import art.unsloth.train as train_mod

    def _patched_calculate_logprobs(lm_head_t, hidden_states, next_input_ids, chunk_size):
        batch_size, seq_len, _ = hidden_states.shape
        log_probs = torch.empty(
            (batch_size, seq_len),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        entropy = torch.empty(
            (batch_size, seq_len),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        lm_head_t = lm_head_t.to(hidden_states.dtype)

        for i in range(0, seq_len, chunk_size):
            chunk_hs = hidden_states[:, i : i + chunk_size, :]
            chunk_input_ids = next_input_ids[:, i : i + chunk_size]
            chunk_logits = torch.matmul(chunk_hs, lm_head_t)
            chunk_selected_logits = torch.gather(
                chunk_logits, dim=-1, index=chunk_input_ids.unsqueeze(-1)
            ).squeeze(-1)
            chunk_logsumexp = torch.logsumexp(chunk_logits, dim=-1)
            log_probs[:, i : i + chunk_size] = chunk_selected_logits - chunk_logsumexp

            # Compute entropy detached from autograd (only used for logging, not loss)
            with torch.no_grad():
                log_probs_full = chunk_logits.detach() - chunk_logsumexp.detach().unsqueeze(-1)
                chunk_entropy = (-torch.exp(log_probs_full) * log_probs_full).sum(dim=-1)
                entropy[:, i : i + chunk_size] = chunk_entropy

            del (
                chunk_hs,
                chunk_input_ids,
                chunk_logits,
                chunk_selected_logits,
                chunk_logsumexp,
                log_probs_full,
                chunk_entropy,
            )
        del hidden_states
        return log_probs, entropy

    train_mod._calculate_logprobs = _patched_calculate_logprobs
    _log("D: _calculate_logprobs — entropy detach + no chunk_size assertion")


# ── Patch E: done_callback guard (backend.py) ────────────────────────


def _patch_done_callback():
    """Guard done_callback to not destroy cached service on error/cancel.

    ART's health monitor (_monitor_openai_server) can time out during
    training pauses → done_callback fires → removes service from cache →
    next _get_service creates new service → loads model again while
    first is still on GPU → OOM.
    """
    from art.local.backend import LocalBackend

    _orig_prepare = LocalBackend._prepare_backend_for_training

    async def _patched_prepare(self, model, config=None):
        base_url, api_key = await _orig_prepare(self, model, config)
        # The original method already set up the monitor task with an
        # unguarded done_callback. We need to find and patch it.
        # Since we can't easily intercept the task after creation,
        # we patch the method to replace the callback behavior.
        return base_url, api_key

    # Direct patch: replace the whole method to add the guard
    import socket
    from typing import cast

    async def _safe_prepare(self, model, config=None):
        from mp_actors import close_proxy

        config_dict = dict(config or {})
        server_args = dict(config_dict.get("server_args", {}))

        if "port" not in server_args:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))
                server_args["port"] = s.getsockname()[1]
        config_dict["server_args"] = server_args
        resolved_config = cast(dict, config_dict)

        service = await self._get_service(model)
        host, port = await service.start_openai_server(config=resolved_config)

        base_url = f"http://{host}:{port}/v1"
        api_key = server_args.get("api_key") or "default"

        def done_callback(task):
            # Only remove the service on clean exit; a monitor timeout/error
            # should NOT destroy the (expensive) cached service.
            if task.cancelled() or task.exception() is not None:
                return
            close_proxy(self._services.pop(model.name))

        asyncio.create_task(
            self._monitor_openai_server(model, base_url, api_key)
        ).add_done_callback(done_callback)

        return base_url, api_key

    LocalBackend._prepare_backend_for_training = _safe_prepare
    _log("E: LocalBackend._prepare_backend_for_training — guarded done_callback")


# ── Patch F: LoRA injection into UnslothState (optional) ─────────────


def _patch_lora_injection():
    """Inject LoRA weights from a previous checkpoint into a fresh adapter.

    Only applied if INJECT_LORA_CHECKPOINT is set.
    """
    if INJECT_LORA_CHECKPOINT is None:
        return

    from art.unsloth.service import UnslothState

    _orig_post_init = getattr(UnslothState, "__post_init__", None)

    original_init = UnslothState.__init__

    def _patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        import os
        from safetensors.torch import load_file

        adapter_path = os.path.join(INJECT_LORA_CHECKPOINT, "adapter_model.safetensors")
        print(f"[art_patches] Injecting LoRA weights from {adapter_path}")
        saved_weights = load_file(adapter_path)
        remapped = {}
        for key, tensor in saved_weights.items():
            model_key = key.replace(
                ".lora_A.weight", ".lora_A.default.weight"
            ).replace(".lora_B.weight", ".lora_B.default.weight")
            remapped[model_key] = tensor
        result = self.peft_model.load_state_dict(remapped, strict=False)
        n_injected = len(remapped) - len(result.unexpected_keys)
        if result.unexpected_keys:
            print(f"[art_patches] WARNING: {len(result.unexpected_keys)} unexpected keys")
        print(
            f"[art_patches] Injected {n_injected}/{len(saved_weights)} LoRA weights "
            f"(missing: {len(result.missing_keys)})"
        )

    UnslothState.__init__ = _patched_init
    _log("F: UnslothState.__init__ — LoRA checkpoint injection")


# ── Patch G (v2): smart-retry timeout on _async_prepare_inputs ─────────

# Optional callable set by the training script so Patch G can read
# phase state when deciding retry vs real deadlock. Shape:
#   () -> {"phase": str, "phase_age_s": float} | None
_phase_getter = None


def register_phase_getter(fn):
    """Training script registers a callable that returns current phase info.

    The patched _prepare_inputs uses it on timeout to distinguish a
    legitimate inter-call wait (phase != model_train) from a real
    in-step hang. Safe to never register — patch falls back to
    conservative behavior (retry up to hard ceiling, then exit).
    """
    global _phase_getter
    _phase_getter = fn


def _patch_deadlock_timeout():
    """Timeout + smart retry on inputs_queue.get() inside _async_prepare_inputs.

    BACKGROUND
    ART's training bridge (art/unsloth/service.py:984-998) has HF Trainer's
    sync `_prepare_inputs` pull from an asyncio.Queue via nest_asyncio
    (nested asyncio.run). Two failure modes exist:

    1. SPURIOUS TIMEOUTS (~93% of historical events, benign). The train_task
       is long-lived across all model.train() calls; HF trainer.train runs
       forever on a 10M-row dataset and calls `_prepare_inputs` in a loop.
       Between model.train() calls, no one is putting on inputs_queue —
       the next get() legitimately waits until the next gather completes
       and the next model.train() starts producing. If that inter-call
       gap (rollouts + judge + gc + checkpoint_delete) exceeds the
       configured timeout, we see a timeout on a completely healthy
       process. Observed: p99 rollouts duration = 431s, max = 560s.

    2. REAL DEADLOCKS (rare, 57-hour hang on 2026-04-17). nest_asyncio +
       asyncio.Queue can lose a wakeup signal between producer/consumer
       when the producer runs on the outer loop and the consumer runs
       on a nested-reentrant invocation. Detectable by: phase stuck at
       "model_train" for >> per-attempt timeout AND no puts have happened
       in that window.

    STRATEGY
    - Per-attempt timeout: ART_DEADLOCK_TIMEOUT_S (default 600s). On timeout,
      collect diagnostics (queue size, time-since-last-put, current phase
      and phase age) and DECIDE:
        * If phase == "model_train" AND phase_age > timeout AND no puts in
          > timeout → genuine hang. Emit "deadlock_exit" event, os._exit(42).
        * Else → spurious (we're between model.train() calls or phase is
          progressing). Emit "timeout_retry" event, loop and re-await.
    - Hard ceiling: ART_DEADLOCK_HARD_CEILING_S (default 1800s total waited
      across all retries). If we blow through it regardless of phase signals,
      assume pathology and exit(42). Belt-and-suspenders.
    - All events are written as JSON lines to ART_DEADLOCK_LOG_PATH. The
      log can be mined later to track real deadlock rate vs spurious rate.

    DOES NOT FIX THE UNDERLYING NEST_ASYNCIO RACE. For that, the
    asyncio.Queue bridge in art.unsloth.service needs to be replaced with
    a threading-based one (trainer.train moved to its own thread). Planned
    as Patch I when/if real deadlock rate warrants the complexity.
    """
    import os
    import sys
    import time
    from datetime import datetime
    from functools import cached_property
    from typing import cast

    from art.unsloth.service import UnslothService

    DEADLOCK_TIMEOUT_S = int(os.environ.get("ART_DEADLOCK_TIMEOUT_S", "600"))
    HARD_CEILING_S = int(os.environ.get("ART_DEADLOCK_HARD_CEILING_S", "1800"))
    LOG_PATH = os.environ.get(
        "ART_DEADLOCK_LOG_PATH", "logs/debug/deadlock_detected.jsonl"
    )

    orig_cached = UnslothService._state
    orig_state_func = orig_cached.func  # type: ignore[attr-defined]

    def _write_event(record: dict) -> None:
        try:
            os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
            with open(LOG_PATH, "a") as f:
                import json
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _patched_state_func(self):
        state = orig_state_func(self)
        inputs_queue = state.inputs_queue
        trainer = state.trainer

        # Wrap put_nowait to record last-put timestamp. process_train_batch
        # is the only producer and calls put_nowait (service.py:119).
        last_put_ts = [time.time()]  # list for closure mutability
        orig_put_nowait = inputs_queue.put_nowait

        def _tracked_put_nowait(item):
            last_put_ts[0] = time.time()
            return orig_put_nowait(item)

        inputs_queue.put_nowait = _tracked_put_nowait  # type: ignore[method-assign]

        def _timeout_prepare_inputs(*_args, **_kwargs):
            async def _get_with_smart_retry():
                t_start = time.time()
                attempt = 0
                while True:
                    try:
                        return await asyncio.wait_for(
                            inputs_queue.get(), timeout=DEADLOCK_TIMEOUT_S
                        )
                    except asyncio.TimeoutError:
                        attempt += 1
                        total_waited = time.time() - t_start
                        since_last_put = time.time() - last_put_ts[0]
                        qsize = getattr(inputs_queue, "qsize", lambda: -1)()
                        step = getattr(
                            getattr(trainer, "state", None), "global_step", None
                        )
                        phase, phase_age = "?", None
                        if _phase_getter is not None:
                            try:
                                info = _phase_getter() or {}
                                phase = info.get("phase", "?")
                                phase_age = info.get("phase_age_s")
                            except Exception:
                                pass

                        # Decide: spurious retry vs real deadlock vs hard ceiling
                        real_deadlock = (
                            phase == "model_train"
                            and (phase_age is not None and phase_age > DEADLOCK_TIMEOUT_S)
                            and since_last_put > DEADLOCK_TIMEOUT_S
                        )
                        over_ceiling = total_waited >= HARD_CEILING_S

                        event = {
                            "schema_version": 1,
                            "patch": "G_v2",
                            "ts": datetime.now().isoformat(),
                            "pid": os.getpid(),
                            "event": (
                                "deadlock_exit" if (real_deadlock or over_ceiling)
                                else "timeout_retry"
                            ),
                            "step": step,
                            "attempt": attempt,
                            "timeout_s": DEADLOCK_TIMEOUT_S,
                            "total_waited_s": round(total_waited, 1),
                            "since_last_put_s": round(since_last_put, 1),
                            "qsize": qsize,
                            "phase": phase,
                            "phase_age_s": (
                                round(phase_age, 1) if phase_age is not None else None
                            ),
                            "reason": (
                                "real_deadlock" if real_deadlock
                                else "hard_ceiling" if over_ceiling
                                else "spurious_healthy_wait"
                            ),
                        }
                        _write_event(event)

                        if real_deadlock or over_ceiling:
                            msg = (
                                f"[ART DEADLOCK] inputs_queue.get() real hang detected: "
                                f"attempt={attempt} total_waited={total_waited:.0f}s "
                                f"phase={phase} phase_age={phase_age} "
                                f"since_last_put={since_last_put:.0f}s reason={event['reason']}. "
                                f"Exiting 42; wrapper will auto-resume from checkpoint."
                            )
                            print(msg, flush=True)
                            os._exit(42)
                        else:
                            msg = (
                                f"[ART TIMEOUT-RETRY] attempt={attempt} "
                                f"total_waited={total_waited:.0f}s phase={phase} "
                                f"phase_age={phase_age} since_last_put={since_last_put:.0f}s "
                                f"— healthy inter-call wait, retrying get()."
                            )
                            print(msg, flush=True)
                            # loop — re-await with a fresh per-attempt timer
                            continue

            return cast(dict, asyncio.run(_get_with_smart_retry()))

        state.trainer._prepare_inputs = _timeout_prepare_inputs
        return state

    new_cached = cached_property(_patched_state_func)
    new_cached.__set_name__(UnslothService, "_state")
    UnslothService._state = new_cached
    _log(
        f"G(v2): _async_prepare_inputs — per-attempt={DEADLOCK_TIMEOUT_S}s "
        f"hard-ceiling={HARD_CEILING_S}s smart-retry + structured telemetry"
    )


# ── Patch H: swallow tokenize_trajectory failures ────────────────────


def _patch_tokenize_safe():
    """Skip trajectories that fail to tokenize instead of crashing the run.

    ART's `tokenize_trajectory` can raise on edge-case trajectories:
    - Qwen3 sometimes emits assistant messages that are only `<think></think>`
      with no content. The chat template strips empty think blocks, leaving
      nothing to render as the "final message" under `continue_final_message=True`.
      Transformers raises ValueError → crashes the whole training step →
      we lose the entire batch's rollouts.
    - Any other template/tokenizer issue in a single trajectory would crash
      all 8 rollouts' gradient update too.

    Observed 2026-04-20 at step 2437 after a Patch-G restart: one rollout
    generated an empty-think-only message, training crashed with rc=1.

    Pattern: treat tokenize failure as "skip this trajectory" (effective
    reward=0 for training purposes). The other 7/8 rollouts in the group
    still contribute their GRPO advantage.
    """
    import art.preprocessing.tokenize as _tok_mod

    _orig_tokenize_trajectory = _tok_mod.tokenize_trajectory
    _counts = {"repair": 0, "drop": 0}

    EMPTY_THINK_RE = re.compile(r"^\s*(<think>\s*</think>)?\s*$")

    def _get(m, key):
        if isinstance(m, dict):
            return m.get(key)
        return getattr(m, key, None)

    def _set_content(m, value):
        if isinstance(m, dict):
            m["content"] = value
        else:
            m.content = value

    def _try_repair(trajectory) -> bool:
        msgs = (
            getattr(trajectory, "messages_and_choices", None)
            or getattr(trajectory, "messages", None)
        )
        if not msgs:
            return False
        for i in range(len(msgs) - 1, -1, -1):
            m = msgs[i]
            if _get(m, "role") != "assistant":
                continue
            content = _get(m, "content") or ""
            if EMPTY_THINK_RE.match(content):
                _set_content(m, " ")
                return True
            return False
        return False

    def _safe_tokenize_trajectory(trajectory, *args, **kwargs):
        try:
            return _orig_tokenize_trajectory(trajectory, *args, **kwargs)
        except Exception as e:
            if _try_repair(trajectory):
                try:
                    result = _orig_tokenize_trajectory(trajectory, *args, **kwargs)
                    _counts["repair"] += 1
                    print(
                        f"[TOKENIZE REPAIR #{_counts['repair']}] empty-final-think "
                        f"patched to ' ' — trajectory retained for training"
                    )
                    return result
                except Exception as e2:
                    e = e2
            _counts["drop"] += 1
            print(
                f"[TOKENIZE DROP #{_counts['drop']}] {type(e).__name__}: "
                f"{str(e)[:200]} — repair failed, dropping trajectory"
            )
            return None

    _tok_mod.tokenize_trajectory = _safe_tokenize_trajectory
    _log("H: tokenize_trajectory — repair empty-final-think, drop only if unrepairable")


# ── Patch I: threading-queue bridge (opt-in, replaces Patch G) ────────


def _patch_threading_bridge():
    """Replace ART's asyncio.Queue + nest_asyncio bridge with threading queues.

    Eliminates the nest_asyncio + asyncio.Queue lost-wakeup race that causes
    rare but real multi-hour hangs (e.g. the 57h hang on 2026-04-17; 3 hangs
    today under external GPU contention).

    Architecture change:
      - inputs_queue: asyncio.Queue → queue.Queue (plain threading).
        Producer (process_train_batch, async context) calls put_nowait —
        thread-safe. Consumer (trainer._prepare_inputs, on the train thread)
        calls get(timeout=...) — blocking in the thread, not the event loop.
      - results_queue: asyncio.Queue → BridgeQueue (queue.Queue subclass with
        coroutine-returning .get()/.join() when called in an asyncio context).
        Producer (trainer.log, on the train thread) calls put_nowait —
        thread-safe. Consumers (`await results_queue.get()` in process_train_batch,
        `await results_queue.join()` in _train_shared) get coroutines via
        run_in_executor wrappers.
      - art.unsloth.train.train: monkey-patched to run trainer.train() via
        `await loop.run_in_executor(None, trainer.train)` instead of calling
        it synchronously. This moves HF trainer onto a real OS thread so the
        threading-queue blocking get() doesn't freeze the event loop.

    Timeouts and the smart-retry logic from Patch G v2 are preserved on the
    consumer side — if threading-queue.get() times out, inspect the registered
    phase snapshot and decide retry vs real-hang exit.

    OPT-IN: set env var ART_USE_THREADING_BRIDGE=1. Mutually exclusive with
    Patch G v2 (only one of them patches _state).
    """
    import os
    import queue as _queue_mod
    import time
    from datetime import datetime
    from functools import cached_property
    from typing import cast

    from art.unsloth.service import UnslothService
    import art.unsloth.train as _train_mod

    DEADLOCK_TIMEOUT_S = int(os.environ.get("ART_DEADLOCK_TIMEOUT_S", "600"))
    HARD_CEILING_S = int(os.environ.get("ART_DEADLOCK_HARD_CEILING_S", "1800"))
    LOG_PATH = os.environ.get(
        "ART_DEADLOCK_LOG_PATH", "logs/debug/deadlock_detected.jsonl"
    )

    class BridgeQueue(_queue_mod.Queue):
        """Thread-safe queue whose .get() and .join() return coroutines when
        called inside an asyncio event loop, staying sync for thread callers.
        put_nowait/task_done are inherited unchanged (sync, thread-safe)."""

        async def _async_get(self):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _queue_mod.Queue.get, self)

        async def _async_join(self):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _queue_mod.Queue.join, self)

        def get(self, block=True, timeout=None):  # type: ignore[override]
            # From asyncio context: return a coroutine (await it).
            # From a non-loop thread: sync blocking get, unchanged.
            try:
                running = asyncio._get_running_loop()
            except Exception:
                running = None
            if running is not None and block and timeout is None:
                return self._async_get()
            return _queue_mod.Queue.get(self, block, timeout)

        def join(self):  # type: ignore[override]
            try:
                running = asyncio._get_running_loop()
            except Exception:
                running = None
            if running is not None:
                return self._async_join()
            return _queue_mod.Queue.join(self)

    orig_cached = UnslothService._state
    orig_state_func = orig_cached.func  # type: ignore[attr-defined]

    def _write_event(record: dict) -> None:
        try:
            os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
            with open(LOG_PATH, "a") as f:
                import json as _json
                f.write(_json.dumps(record) + "\n")
        except Exception:
            pass

    def _patched_state_func(self):
        state = orig_state_func(self)
        # Replace the queues with threading variants.
        t_inputs: _queue_mod.Queue = _queue_mod.Queue()
        t_results: BridgeQueue = BridgeQueue()
        state.inputs_queue = t_inputs
        state.results_queue = t_results
        trainer = state.trainer

        # Track last-put timestamp for diagnostics (same as Patch G v2).
        last_put_ts = [time.time()]
        orig_put_nowait = t_inputs.put_nowait

        def _tracked_put_nowait(item):
            last_put_ts[0] = time.time()
            return orig_put_nowait(item)

        t_inputs.put_nowait = _tracked_put_nowait  # type: ignore[method-assign]

        def _sync_prepare_inputs(*_args, **_kwargs):
            """Sync blocking read from threading inputs_queue — runs on the
            trainer thread. On timeout, apply Patch G v2's smart-retry logic."""
            t_start = time.time()
            attempt = 0
            while True:
                try:
                    return t_inputs.get(timeout=DEADLOCK_TIMEOUT_S)
                except _queue_mod.Empty:
                    pass
                attempt += 1
                total_waited = time.time() - t_start
                since_last_put = time.time() - last_put_ts[0]
                step = getattr(
                    getattr(trainer, "state", None), "global_step", None
                )
                phase, phase_age = "?", None
                if _phase_getter is not None:
                    try:
                        info = _phase_getter() or {}
                        phase = info.get("phase", "?")
                        phase_age = info.get("phase_age_s")
                    except Exception:
                        pass

                real_deadlock = (
                    phase == "model_train"
                    and (phase_age is not None and phase_age > DEADLOCK_TIMEOUT_S)
                    and since_last_put > DEADLOCK_TIMEOUT_S
                )
                over_ceiling = total_waited >= HARD_CEILING_S

                event = {
                    "schema_version": 1,
                    "patch": "I",
                    "ts": datetime.now().isoformat(),
                    "pid": os.getpid(),
                    "event": (
                        "deadlock_exit" if (real_deadlock or over_ceiling)
                        else "timeout_retry"
                    ),
                    "bridge": "threading",
                    "step": step,
                    "attempt": attempt,
                    "timeout_s": DEADLOCK_TIMEOUT_S,
                    "total_waited_s": round(total_waited, 1),
                    "since_last_put_s": round(since_last_put, 1),
                    "qsize": t_inputs.qsize(),
                    "phase": phase,
                    "phase_age_s": (
                        round(phase_age, 1) if phase_age is not None else None
                    ),
                    "reason": (
                        "real_deadlock" if real_deadlock
                        else "hard_ceiling" if over_ceiling
                        else "spurious_healthy_wait"
                    ),
                }
                _write_event(event)

                if real_deadlock or over_ceiling:
                    print(
                        f"[ART DEADLOCK(bridge=threading)] real hang: attempt={attempt} "
                        f"waited={total_waited:.0f}s phase={phase} phase_age={phase_age} "
                        f"since_put={since_last_put:.0f}s reason={event['reason']}. Exit 42.",
                        flush=True,
                    )
                    os._exit(42)
                else:
                    print(
                        f"[ART TIMEOUT-RETRY(bridge=threading)] attempt={attempt} "
                        f"waited={total_waited:.0f}s phase={phase} since_put={since_last_put:.0f}s "
                        f"— healthy wait, retrying get().",
                        flush=True,
                    )
                    # loop continues with a fresh per-attempt timer

        trainer._prepare_inputs = _sync_prepare_inputs
        return state

    new_cached = cached_property(_patched_state_func)
    new_cached.__set_name__(UnslothService, "_state")
    UnslothService._state = new_cached

    # Also patch art.unsloth.train.train to run trainer.train on a real OS
    # thread so threading-queue blocking get() in _prepare_inputs doesn't
    # freeze the event loop.
    _orig_train_fn = _train_mod.train

    async def _thread_bridged_train(trainer, results_queue):
        from collections import defaultdict
        _compute_loss = trainer.compute_loss
        _log = trainer.log
        trainer.compute_loss = _train_mod.get_compute_loss_fn(trainer)
        trainer.log = _train_mod.get_log_fn(trainer, results_queue)  # type: ignore[method-assign]
        try:
            is_dict = isinstance(getattr(trainer, "_metrics", None), dict)
            is_train_dict = is_dict and isinstance(
                trainer._metrics.get("train"), dict
            )
        except Exception:
            is_train_dict = False
        if not is_train_dict:
            trainer._metrics = {"train": defaultdict(list)}
        try:
            loop = asyncio.get_event_loop()
            # Run blocking HF trainer.train on the default thread pool. The
            # event loop stays free to run rollouts / process_train_batch
            # while the thread is blocked in _sync_prepare_inputs.
            await loop.run_in_executor(None, trainer.train)
        finally:
            trainer.compute_loss = _compute_loss
            trainer.log = _log  # type: ignore[method-assign]

    _train_mod.train = _thread_bridged_train

    _log(
        f"I: threading-bridge — inputs_queue=queue.Queue, results_queue=BridgeQueue, "
        f"trainer.train=on-thread. timeout={DEADLOCK_TIMEOUT_S}s ceiling={HARD_CEILING_S}s"
    )


# ── Apply all patches on import ───────────────────────────────────────


def apply_all():
    """Apply all patches. Safe to call multiple times."""
    if _APPLIED:
        return
    _patch_calculate_logprobs()
    _patch_done_callback()
    _patch_lora_injection()
    # Patch I and Patch G v2 both hook _state — mutually exclusive.
    use_threading = os.environ.get("ART_USE_THREADING_BRIDGE", "0") == "1"
    if use_threading:
        _patch_threading_bridge()
    else:
        _patch_deadlock_timeout()
    _patch_tokenize_safe()
    print(f"[art_patches] All {len(_APPLIED)} patches applied successfully")


import os  # noqa: E402 — used in apply_all's env check
apply_all()
