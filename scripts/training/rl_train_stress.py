"""Stress-test harness for ART 0.5.17 deadlock reproduction.

Mirrors scripts/training/rl_train.py's ART usage (backend, model registration,
rollout + gather + train loop, Patch G) but with:
  - Qwen2.5-0.5B-Instruct instead of Qwen3-14B
  - Random rewards instead of Gemini judge
  - Synthetic one-turn scenarios (no calendar env, no tools)
  - Short max tokens

Goal: run ART's real code paths (vLLM + Unsloth + queue bridge + nest_asyncio)
at ~10× iteration rate to measure deadlock cadence and validate patches.

Output dir: runs/rl_stress_qwen25_05b/
"""

import asyncio
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Apply ART patches (Patch G deadlock timeout, Patch H tokenize skip) before art import
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import calendar_agent.art_patches  # noqa: F401

import art
from art import dev
from art.local import LocalBackend
from art.utils import iterate_dataset
from openai import AsyncOpenAI

# ── Config ───────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
RUN_DIR = Path("runs/rl_stress_qwen25_05b")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Small synthetic scenario set
NUM_SCENARIOS = 30
ROLLOUTS_PER_GROUP = 8            # match real training (async concurrency)
GROUPS_PER_STEP = 1
MAX_TOKENS = 128
NUM_EPOCHS = 10000  # effectively unbounded — we're measuring deadlock rate


# ── Rollout ──────────────────────────────────────────────────

async def rollout(model: art.Model, scenario: dict) -> art.Trajectory:
    """Single-turn rollout with random reward."""
    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )
    messages = [
        {"role": "user", "content": scenario["prompt"]},
    ]
    response = await client.chat.completions.create(
        model=model.get_inference_name(),
        messages=messages,
        temperature=0.9,
        max_completion_tokens=MAX_TOKENS,
    )
    choice = response.choices[0]
    # Random continuous reward — drops GRPO skip rate to ~0 so every step
    # exercises the training path (more _prepare_inputs calls → more race
    # opportunities for deadlock observation).
    reward = float(random.uniform(0.0, 1.0))
    return art.Trajectory(
        messages_and_choices=[*messages, choice],
        reward=reward,
        metrics={"correct": reward},
    )


# ── Main ─────────────────────────────────────────────────────

async def main():
    print(f"[stress] start pid={os.getpid()} ts={datetime.now().isoformat()}")

    model = art.TrainableModel(
        name="stress-001",
        project="rl-stress",
        base_model=BASE_MODEL,
        _internal_config=dev.InternalModelConfig(
            init_args=dev.InitArgs(
                load_in_4bit=True,
                max_lora_rank=32,
            ),
            engine_args=dev.EngineArgs(
                max_model_len=1024,
                max_num_batched_tokens=1024,
                max_num_seqs=8,
                gpu_memory_utilization=0.70,
                enforce_eager=True,
                enable_sleep_mode=True,
                quantization="bitsandbytes",
                load_format="bitsandbytes",
            ),
            trainer_args=dev.TrainerArgs(
                per_device_train_batch_size=4,   # match real training
                gradient_accumulation_steps=1,
                logging_steps=1,
                num_generations=4,               # match real training
                max_completion_length=MAX_TOKENS,
                max_grad_norm=0.1,
                optim="adamw_torch",
                bf16=True,
                fp16=False,
            ),
        ),
    )

    backend = LocalBackend(in_process=True)
    from art.dev.openai_server import OpenAIServerConfig, ServerArgs
    await model.register(
        backend,
        _openai_client_config=OpenAIServerConfig(
            server_args=ServerArgs(port=8007),
        ),
    )

    # Synthetic scenarios — trivial prompts the model can respond to in a few tokens
    scenarios = [
        {"prompt": f"In one sentence, describe the number {i}."}
        for i in range(NUM_SCENARIOS)
    ]
    random.shuffle(scenarios)

    training_iterator = iterate_dataset(
        scenarios,
        groups_per_step=GROUPS_PER_STEP,
        num_epochs=NUM_EPOCHS,
        initial_step=await model.get_step(),
    )

    step_wall_times = []
    total_start = time.time()

    for batch in training_iterator:
        step_t0 = time.time()
        groups = [
            art.TrajectoryGroup(
                rollout(model, batch.items[i])
                for _ in range(ROLLOUTS_PER_GROUP)
            )
            for i in range(len(batch.items))
        ]
        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc=f"stress step {batch.step}",
            max_exceptions=ROLLOUTS_PER_GROUP * len(batch.items),
        )

        rewards = [t.reward for g in finished_groups for t in g.trajectories]
        mean_r = sum(rewards) / max(1, len(rewards))

        await model.delete_checkpoints()
        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=1e-5),
        )

        step_dt = time.time() - step_t0
        step_wall_times.append(step_dt)
        elapsed_min = (time.time() - total_start) / 60
        print(
            f"[stress] step={batch.step} epoch={batch.epoch} "
            f"wall={step_dt:5.1f}s mean_reward={mean_r:.2f} "
            f"total_elapsed={elapsed_min:5.1f}min",
            flush=True,
        )

    await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
