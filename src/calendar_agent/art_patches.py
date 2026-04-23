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


# ── Patch G: deadlock timeout on _async_prepare_inputs ────────────────


def _patch_deadlock_timeout():
    """Add a timeout to inputs_queue.get() inside _async_prepare_inputs.

    ART's training bridge (service.py:988-996) has HF Trainer's sync
    `_prepare_inputs` call into an asyncio.Queue via nest_asyncio. Under
    race conditions the wakeup doesn't propagate and the get() blocks
    forever. Observed in practice at steps 112, 1640, 2325, 2437 of our
    training — same stack each time.

    This patch wraps the get() in asyncio.wait_for(timeout=300s). On
    timeout, we've confirmed the process is deadlocked (normal step is
    5-15s); we os._exit(42) so the outer shell can restart. ART
    auto-resumes from the last saved checkpoint.

    Mirrors the pattern used upstream in PR #429 (OpenPipe/ART) which
    applies the same timeout+recover treatment to results_queue.join()
    (a sibling deadlock site in the same queue protocol).
    """
    import os
    import sys
    from datetime import datetime
    from functools import cached_property
    from typing import cast

    from art.unsloth.service import UnslothService

    # Configurable via ART_DEADLOCK_TIMEOUT_S. Default 300s for real training
    # (normal step 5-15s → 20-60× margin). Stress harness sets 30s for faster
    # iteration.
    DEADLOCK_TIMEOUT_S = int(os.environ.get("ART_DEADLOCK_TIMEOUT_S", "300"))

    orig_cached = UnslothService._state
    # cached_property exposes the underlying function as .func
    orig_state_func = orig_cached.func  # type: ignore[attr-defined]

    # Log path configurable via ART_DEADLOCK_LOG_PATH. Default keeps legacy
    # relative path for backward compat.
    LOG_PATH = os.environ.get(
        "ART_DEADLOCK_LOG_PATH", "logs/debug/deadlock_detected.jsonl"
    )

    def _patched_state_func(self):
        state = orig_state_func(self)
        inputs_queue = state.inputs_queue
        trainer = state.trainer

        def _timeout_prepare_inputs(*_args, **_kwargs):
            async def _get_with_timeout():
                try:
                    return await asyncio.wait_for(
                        inputs_queue.get(), timeout=DEADLOCK_TIMEOUT_S
                    )
                except asyncio.TimeoutError:
                    # Best-effort step number (may not exist if deadlock pre-train)
                    step = getattr(getattr(trainer, "state", None), "global_step", None)
                    ts = datetime.now().isoformat()
                    record = (
                        f'{{"ts": "{ts}", "pid": {os.getpid()}, '
                        f'"step": {step!r}, "timeout_s": {DEADLOCK_TIMEOUT_S}}}'
                    )
                    msg = (
                        f"[ART DEADLOCK] _async_prepare_inputs: inputs_queue.get() "
                        f"timed out after {DEADLOCK_TIMEOUT_S}s at step={step} "
                        f"(known race in ART's queue bridge, see PR #429). "
                        f"Exiting 42; ART will auto-resume from last checkpoint."
                    )
                    print(msg, flush=True)
                    try:
                        os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
                        with open(LOG_PATH, "a") as f:
                            f.write(record + "\n")
                    except Exception:
                        pass
                    # Hard exit — can't safely recover from inside a deadlocked
                    # nested asyncio loop.
                    os._exit(42)

            return cast(dict, asyncio.run(_get_with_timeout()))

        state.trainer._prepare_inputs = _timeout_prepare_inputs
        return state

    new_cached = cached_property(_patched_state_func)
    # cached_property needs __set_name__ to know the attribute name; by
    # assigning to a class attr after class creation, we must call it manually.
    new_cached.__set_name__(UnslothService, "_state")
    UnslothService._state = new_cached
    _log(f"G: _async_prepare_inputs — timeout={DEADLOCK_TIMEOUT_S}s + exit(42) on queue deadlock")


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
    _fail_count = {"n": 0}

    def _safe_tokenize_trajectory(*args, **kwargs):
        try:
            return _orig_tokenize_trajectory(*args, **kwargs)
        except Exception as e:
            _fail_count["n"] += 1
            print(
                f"[TOKENIZE SKIP #{_fail_count['n']}] {type(e).__name__}: "
                f"{str(e)[:200]} — dropping trajectory, continuing training"
            )
            return None

    _tok_mod.tokenize_trajectory = _safe_tokenize_trajectory
    _log("H: tokenize_trajectory — swallow exceptions (skip bad trajectory, continue)")


# ── Apply all patches on import ───────────────────────────────────────


def apply_all():
    """Apply all patches. Safe to call multiple times."""
    if _APPLIED:
        return
    _patch_calculate_logprobs()
    _patch_done_callback()
    _patch_lora_injection()
    _patch_deadlock_timeout()
    _patch_tokenize_safe()
    print(f"[art_patches] All {len(_APPLIED)} patches applied successfully")


apply_all()
