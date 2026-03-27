"""Runtime monkey-patches for ART 0.5.4 + Unsloth 2025.2.15.

Apply these before importing ART to fix:
- Async deadlock between service.py and HF Trainer (patches A+C)
- OOM from service cache destruction on health monitor timeout (patch E)
- OOM from entropy in autograd graph (patch D)
- Assertion error on non-divisible sequence lengths (patch D)
- Missing gc/empty_cache between inference and training (patch B)

Usage:
    import calendar_agent.art_patches  # patches applied on import
    import art  # now safe to use
"""

import asyncio
import functools
import gc
import queue
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, AsyncGenerator, cast

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


# ── Patch A: queue.Queue + sync _prepare_inputs (state.py) ────────────


def _patch_model_state_init():
    """Replace asyncio.Queue with queue.Queue in ModelState.__init__.

    asyncio.Queue deadlocks when producer and consumer share the same
    event loop via nest_asyncio. queue.Queue is thread-safe and blocks
    the trainer thread (which runs via run_in_executor) without blocking
    the event loop.
    """
    from art.unsloth.state import ModelState

    _orig_init = ModelState.__init__

    @functools.wraps(_orig_init)
    def _patched_init(self, config):
        _orig_init(self, config)

        # Inject LoRA weights from a previous checkpoint if configured
        if INJECT_LORA_CHECKPOINT is not None:
            import os
            from safetensors.torch import load_file

            adapter_path = os.path.join(INJECT_LORA_CHECKPOINT, "adapter_model.safetensors")
            print(f"[art_patches] Injecting LoRA weights from {adapter_path}")
            saved_weights = load_file(adapter_path)
            # Remap keys: PEFT adds ".default." namespace for the default adapter
            remapped = {}
            for key, tensor in saved_weights.items():
                model_key = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
                remapped[model_key] = tensor
            # Use load_state_dict with strict=False to inject LoRA weights
            result = self.peft_model.load_state_dict(remapped, strict=False)
            n_injected = len(remapped) - len(result.unexpected_keys)
            if result.unexpected_keys:
                print(f"[art_patches] WARNING: {len(result.unexpected_keys)} unexpected keys")
            print(f"[art_patches] Injected {n_injected}/{len(saved_weights)} LoRA weights (missing: {len(result.missing_keys)})")

        # Override asyncio.Queue with thread-safe queue.Queue
        self.inputs_queue = queue.Queue()

        # Patch trainer _prepare_inputs to block synchronously
        def _sync_prepare_inputs(*_, **__):
            inputs = self.inputs_queue.get()
            return cast(dict, inputs)

        self.trainer._prepare_inputs = _sync_prepare_inputs

    ModelState.__init__ = _patched_init
    _log("A: ModelState.__init__ — queue.Queue + sync _prepare_inputs")


# ── Patch B: train_mode gc/empty_cache/sleep (state.py) ───────────────


def _patch_train_mode():
    """Add gc.collect + empty_cache between inference pause and training yield.

    Without this, inference activations remain on GPU during training,
    causing OOM on 12 GiB cards.
    """
    from art.unsloth.state import vLLMState
    from art.unsloth.train import gc_and_empty_cuda_cache

    @asynccontextmanager
    async def _patched_train_mode(self) -> AsyncGenerator[None, None]:
        if not self.enable_sleep_mode:
            yield
            return
        try:
            await self.pause_engine()
            try:
                if self.async_engine.engine.has_unfinished_requests():
                    await self.async_engine.sleep(level=1)
                else:
                    await self.async_engine.reset_prefix_cache()
                    await self.async_engine.sleep(level=2)
                gc_and_empty_cuda_cache()
                yield
            finally:
                gc_and_empty_cuda_cache()
                await asyncio.sleep(0.1)
                await self.async_engine.wake_up()
        finally:
            await self.resume_engine()

    vLLMState.train_mode = _patched_train_mode
    _log("B: vLLMState.train_mode — gc/empty_cache between inference and training")


# ── Patch C: run_in_executor + call_soon_threadsafe (train.py) ────────


def _patch_train_and_log():
    """Run trainer.train() in executor thread; use call_soon_threadsafe for log.

    Without run_in_executor, queue.Queue.get() in _prepare_inputs blocks
    the event loop (since trainer.train is sync). With run_in_executor,
    it blocks only the thread, keeping the loop free to produce items.

    Without call_soon_threadsafe, putting results on the asyncio.Queue
    from the executor thread is not thread-safe.
    """
    import art.unsloth.train as train_mod
    from collections import defaultdict

    async def _patched_train(trainer, results_queue):
        _compute_loss = trainer.compute_loss
        _log_fn = trainer.log
        trainer.compute_loss = train_mod.get_compute_loss_fn(trainer)
        trainer.log = train_mod.get_log_fn(trainer, results_queue)
        try:
            is_dict = isinstance(getattr(trainer, "_metrics", None), dict)
            is_train_dict = is_dict and isinstance(trainer._metrics.get("train"), dict)
        except Exception:
            is_train_dict = False
        if not is_train_dict:
            trainer._metrics = {"train": defaultdict(list)}
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, trainer.train)
        finally:
            trainer.compute_loss = _compute_loss
            trainer.log = _log_fn

    _orig_get_log_fn = train_mod.get_log_fn

    def _patched_get_log_fn(trainer, results_queue):
        loop = asyncio.get_event_loop()

        def log(logs, start_time=None):
            metrics = {
                key: sum(val) / len(val)
                for key, val in trainer._metrics["train"].items()
            }
            if next(iter(logs.keys())).startswith("eval_"):
                metrics = {f"eval_{key}": val for key, val in metrics.items()}
            logs = {**logs, **metrics}
            logs.pop("learning_rate", None)
            loop.call_soon_threadsafe(results_queue.put_nowait, logs)
            trainer._metrics["train"].clear()

        return log

    train_mod.train = _patched_train
    train_mod.get_log_fn = _patched_get_log_fn
    _log("C: train() — run_in_executor; get_log_fn — call_soon_threadsafe")


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

    ART's health monitor (_monitor_openai_server) times out during
    training pauses → done_callback fires → removes service from cache →
    next _get_service creates new ModelState → loads model again while
    first is still on GPU → OOM.
    """
    from art.local.backend import LocalBackend

    _orig_prepare = LocalBackend._prepare_backend_for_training

    async def _patched_prepare(self, model, config=None):
        from mp_actors import close_proxy

        service = await self._get_service(model)
        await service.start_openai_server(config=config)
        server_args = (config or {}).get("server_args", {})

        base_url = f"http://{server_args.get('host', '0.0.0.0')}:{server_args.get('port', 8000)}/v1"
        api_key = server_args.get("api_key", None) or "default"

        def done_callback(task):
            # Only remove the service on clean exit; a monitor timeout/error
            # should NOT destroy the (expensive) cached ModelState.
            if task.cancelled() or task.exception() is not None:
                return
            close_proxy(self._services.pop(model.name))

        asyncio.create_task(
            self._monitor_openai_server(model.name, base_url, api_key)
        ).add_done_callback(done_callback)

        return base_url, api_key

    LocalBackend._prepare_backend_for_training = _patched_prepare
    _log("E: LocalBackend._prepare_backend_for_training — guarded done_callback")


# ── Apply all patches on import ───────────────────────────────────────

def apply_all():
    """Apply all patches. Safe to call multiple times."""
    if _APPLIED:
        return
    _patch_model_state_init()
    _patch_train_mode()
    _patch_train_and_log()
    _patch_calculate_logprobs()
    _patch_done_callback()
    print(f"[art_patches] All {len(_APPLIED)} patches applied successfully")


apply_all()
