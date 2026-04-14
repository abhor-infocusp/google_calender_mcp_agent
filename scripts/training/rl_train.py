import asyncio
import gc
import json
import logging
import os
import random
import sys
import time
import traceback
from datetime import datetime

logging.basicConfig(
    format="%(asctime)s %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["CC"] = "/usr/bin/gcc"  # Triton needs this in spawned subprocesses
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import ipykernel.iostream as _io

    _orig_outstream_close = _io.OutStream.close

    def _safe_outstream_close(self):
        if getattr(self, "watch_fd_thread", None) is None:
            return
        return _orig_outstream_close(self)

    _io.OutStream.close = _safe_outstream_close
except ImportError:
    pass

import calendar_agent.art_patches  # noqa: F401 — must be before art imports

# No LoRA injection — starting fresh from SFT v5 baseline (74.6%)
# calendar_agent.art_patches.INJECT_LORA_CHECKPOINT = "rl_runs/single_category_modifier_correction/checkpoint"

import art
import torch
import vertexai
from art import dev
from art.local import LocalBackend
from art.utils import iterate_dataset
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt
from vertexai.generative_models import GenerativeModel

from calendar_agent.environment import CalendarEnvironment
from calendar_agent.core import (
    compute_fallback_now,
    dispatch_tool_call,
    filter_by_days,
    format_tool_result,
    snapshot_events,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, format_day_state_text
from calendar_agent.paths import RL_DATA_DIR as _RL_DATA_DIR, RL_JSON_CALENDAR_DIR, RL_QUERY_DIR, CREDENTIALS_PATH
from calendar_agent.tools import get_openai_tools_minimal

random.seed(42)

# ── Paths ──────────────────────────────────────────────────

JSON_CALENDAR_DIR = str(RL_JSON_CALENDAR_DIR)
QUERY_DIR = str(RL_QUERY_DIR)

# ── Configuration ──────────────────────────────────────────

MAX_TURNS = 5  # cap agent turns to bound trajectory length


# ── Data Models ────────────────────────────────────────────


class CalendarScenario(BaseModel):
    """A flattened training scenario: one calendar + one query."""

    id: str
    calendar_index: int
    query_index: int
    query: str
    expected_behavior: str
    category: str
    complexity: str
    addressed_days: list[str]
    current_time: str
    calendar_file_path: str


class ProjectTrajectory(art.Trajectory):
    final_answer_text: str | None = None


class CalendarRolloutInput(BaseModel):
    step: int
    scenario: CalendarScenario


# ── Data Loading ───────────────────────────────────────────


def load_all_scenarios() -> list[CalendarScenario]:
    """Load all 50 calendar/query pairs and flatten into individual scenarios."""
    scenarios = []

    for cal_index in range(50):
        cal_path = os.path.join(JSON_CALENDAR_DIR, f"{cal_index}.txt")
        query_path = os.path.join(QUERY_DIR, f"{cal_index}.txt")

        if not os.path.exists(cal_path) or not os.path.exists(query_path):
            continue

        fallback_now = compute_fallback_now(cal_path)

        with open(query_path) as f:
            queries = json.load(f)

        for q_index, q in enumerate(queries):
            current_time = q.get("current_time", "")
            if current_time:
                current_time = current_time.replace("T", " ")
            else:
                current_time = fallback_now

            scenarios.append(
                CalendarScenario(
                    id=f"cal_{cal_index}_q_{q_index}",
                    calendar_index=cal_index,
                    query_index=q_index,
                    query=q["query"],
                    expected_behavior=q.get("expected_behavior", ""),
                    category=q.get("category", "Unknown"),
                    complexity=q.get("complexity", "Unknown"),
                    addressed_days=q.get("addressed_days", []),
                    current_time=current_time,
                    calendar_file_path=os.path.abspath(cal_path),
                )
            )

    return scenarios


OPENAI_TOOLS = get_openai_tools_minimal(include_final_answer=True)

# ── Evaluation (uses Gemini judge via Vertex AI) ──────────

# Load Google credentials from file
CREDENTIALS_PATH = str(CREDENTIALS_PATH)

_gcp_credentials = None
if os.path.exists(CREDENTIALS_PATH):
    from google.oauth2.credentials import Credentials as OAuth2Credentials

    with open(CREDENTIALS_PATH) as _f:
        _cred_data = json.load(_f)
    _gcp_credentials = OAuth2Credentials(
        token=None,
        refresh_token=_cred_data["refresh_token"],
        client_id=_cred_data["client_id"],
        client_secret=_cred_data["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )

# Initialize Vertex AI for the Gemini evaluator
vertexai.init(
    project=os.environ.get("GCP_PROJECT", "internal-ml-exp"),
    location=os.environ.get("GCP_LOCATION", "us-central1"),
    credentials=_gcp_credentials,
)

eval_model = GenerativeModel(
    "gemini-2.0-flash-001",
    system_instruction=[EVAL_SYSTEM_PROMPT],
)


@retry(stop=stop_after_attempt(3))
async def evaluate_trajectory(
    query: str,
    final_output: str,
    expected: str,
    before_days: dict,
    after_days: dict,
) -> str:
    """Ask Gemini to evaluate whether the trajectory was correct.

    Returns one of: 'Correct', 'Incorrect', 'Unsure'.
    """
    before_text = format_day_state_text(before_days)
    after_text = format_day_state_text(after_days)

    prompt = f"""\
Query: {query}

Response: {final_output if final_output else '(no response)'}

Expected: {expected if expected else '(not specified)'}

Before:
{before_text}

After:
{after_text}

Was the task completed correctly? End with one word: Correct or Incorrect."""

    try:
        response = await asyncio.to_thread(eval_model.generate_content, prompt)
        verdict_text = response.text.strip()
        lines = [l.strip() for l in verdict_text.splitlines() if l.strip()]
        # Scan from last line for exact verdict
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if line_lower == token.lower():
                    return token
        # Fallback: substring scan
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if token.lower() in line_lower:
                    return token
        return "Incorrect"
    except Exception as e:
        print(f"[EVAL ERROR] {e}")
        return "Incorrect"


# ── Rollout ────────────────────────────────────────────────

async def rollout(
    model: art.Model, calendar_input: CalendarRolloutInput
) -> ProjectTrajectory:
    scenario = calendar_input.scenario

    # Fresh environment per rollout
    env = CalendarEnvironment()
    events = CalendarEnvironment.load_json_calendar(scenario.calendar_file_path)
    env.initialize(events=events, now=scenario.current_time)

    # Snapshot BEFORE state
    before_snap = snapshot_events(env)

    traj = ProjectTrajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": scenario.id,
            "step": calendar_input.step,
            "category": scenario.category,
            "complexity": scenario.complexity,
        },
    )

    # No system prompt — model learns behavior from SFT data
    traj.messages_and_choices = [
        {"role": "user", "content": scenario.query},
    ]

    traj.tools = OPENAI_TOOLS

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    final_answer_text = None
    num_turns = 0
    num_tool_calls = 0
    had_error = False
    tool_names_list = []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    # ── Agent loop ──
    for _ in range(MAX_TURNS):
        num_turns += 1
        try:
            response = await client.chat.completions.create(
                model=model.get_inference_name(),
                temperature=1,
                messages=traj.messages(),
                tools=traj.tools,
            )
        except Exception as e:
            err_str = str(e)
            if "maximum context length" in err_str:
                print(f"  [ROLLOUT {scenario.id}] Context overflow at turn {_}, ending trajectory")
                traj.metrics["context_overflow"] = 1.0
            else:
                print(f"[ROLLOUT ERROR] LLM call failed: {e}")
                traceback.print_exc()
            had_error = True
            break

        # Track token usage
        if response.usage:
            total_prompt_tokens += response.usage.prompt_tokens or 0
            total_completion_tokens += response.usage.completion_tokens or 0

        response_message = response.choices[0].message
        traj.messages_and_choices.append(response.choices[0])

        # Debug: log what model generates
        tc_names = [tc.function.name for tc in (response_message.tool_calls or [])]
        content_preview = (response_message.content or "")[:100]
        n_tools = len(traj.tools) if traj.tools else 0
        n_msgs = len(traj.messages())
        model_name = model.get_inference_name()
        print(f"  [DEBUG {scenario.id}] turn={_} tool_calls={tc_names} content='{content_preview}' finish={response.choices[0].finish_reason} n_tools={n_tools} n_msgs={n_msgs} model={model_name}")

        if not response_message.tool_calls:
            if response_message.content:
                final_answer_text = response_message.content
            break

        hit_final_answer = False
        try:
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                if tool_name == "return_final_answer":
                    final_answer_text = tool_args.get("answer", "")
                    traj.messages_and_choices.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": final_answer_text,
                        }
                    )
                    hit_final_answer = True
                    break

                num_tool_calls += 1
                tool_names_list.append(tool_name)

                # Dispatch to CalendarEnvironment via run_trajectory helper
                result = dispatch_tool_call(env, tool_name, tool_args)
                result_str = format_tool_result(result)
                traj.messages_and_choices.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": result_str,
                    }
                )
        except Exception as e:
            print(f"Error executing tool call: {e}")
            had_error = True
            break

        if hit_final_answer:
            break

    # ── Compute shaped reward ──
    # Collect tool names used during the trajectory
    tool_names_used = []
    for msg in traj.messages_and_choices:
        if isinstance(msg, dict) and msg.get("role") == "tool":
            name = msg.get("name", "")
            if name and name != "return_final_answer":
                tool_names_used.append(name)

    # Binary reward: 0.0 or 1.0 only. No shaped intermediates — forces the
    # model to optimise for correctness rather than gaming tool-use patterns.
    reward = 0.0
    verdict = "NoAnswer"

    judge_latency_s = 0.0
    if final_answer_text is not None:
        # Run Gemini judge for full correctness
        after_snap = snapshot_events(env)
        before_days = filter_by_days(before_snap, scenario.addressed_days)
        after_days = filter_by_days(after_snap, scenario.addressed_days)

        judge_start = time.monotonic()
        verdict = await evaluate_trajectory(
            query=scenario.query,
            final_output=final_answer_text,
            expected=scenario.expected_behavior,
            before_days=before_days,
            after_days=after_days,
        )
        judge_latency_s = round(time.monotonic() - judge_start, 2)

        if verdict == "Correct":
            reward = 1.0

    traj.reward = reward
    traj.metrics["correct"] = 1.0 if verdict == "Correct" else 0.0
    traj.metrics["verdict"] = {"Correct": 1, "Incorrect": 0}.get(verdict, -1)
    traj.metrics["shaped_reward"] = reward
    traj.metrics["num_turns"] = float(num_turns)
    traj.metrics["num_tool_calls"] = float(num_tool_calls)
    traj.metrics["had_error"] = 1.0 if had_error else 0.0
    traj.metrics["no_final_answer"] = 1.0 if final_answer_text is None else 0.0
    traj.metadata["tool_names"] = ",".join(tool_names_list) if tool_names_list else ""
    traj.metrics["judge_latency_s"] = judge_latency_s
    traj.metrics["prompt_tokens"] = float(total_prompt_tokens)
    traj.metrics["completion_tokens"] = float(total_completion_tokens)
    traj.final_answer_text = final_answer_text
    print(
        f"  [DEBUG {scenario.id}] verdict={verdict} reward={reward} "
        f"tools={tool_names_used} final_answer='{(final_answer_text or '')[:100]}'"
    )

    return traj


# ── Training Utilities ─────────────────────────────────────


def print_gpu_usage(label: str = ""):
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(
            f"  [GPU {label}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB | Total: {total:.2f} GB"
        )


def gpu_snapshot(label: str = "") -> dict:
    """Return GPU memory stats as a dict and print them."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free, total_mem = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    snap = {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "max_allocated_gb": round(max_allocated, 3),
        "total_gb": round(total, 3),
        "free_gb": round(free_gib, 3),
    }
    if label:
        print(f"  [GPU {label}] alloc={allocated:.2f} res={reserved:.2f} free={free_gib:.2f} max_alloc={max_allocated:.2f} total={total:.2f}")
    return snap


class StepTimer:
    """Simple wall-clock timer for named phases within a training step."""

    def __init__(self):
        self.records: dict[str, float] = {}
        self._current_phase: str | None = None
        self._phase_start: float = 0.0

    def start(self, phase: str):
        self.stop()
        self._current_phase = phase
        self._phase_start = time.monotonic()

    def stop(self):
        if self._current_phase is not None:
            elapsed = time.monotonic() - self._phase_start
            self.records[self._current_phase] = round(elapsed, 2)
            self._current_phase = None

    def get(self) -> dict[str, float]:
        self.stop()
        return dict(self.records)


# ── Main ───────────────────────────────────────────────────


async def main():
    # ── Model & Backend ──
    model = art.TrainableModel(
        name="calendar-agent-001",
        project="calendar-agent",
        # SFT base model. RL LoRA weights from M&C run are injected via
        # art_patches.INJECT_LORA_CHECKPOINT after fresh LoRA creation.
        base_model="sft_output/merged_tmp",
        _internal_config=dev.InternalModelConfig(
            init_args=dev.InitArgs(
                load_in_4bit=True,
                use_bitsandbytes=False,
                max_lora_rank=32,
            ),
            engine_args=dev.EngineArgs(
                max_model_len=3076,
                max_num_batched_tokens=3076,
                max_num_seqs=4,
                gpu_memory_utilization=0.85,
                enforce_eager=True,
                swap_space=2,
                enable_sleep_mode=True,
            ),
            trainer_args=dev.TrainerArgs(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                num_generations=2,
                max_completion_length=512,
                max_grad_norm=0.1,
                optim="paged_adamw_8bit",
                bf16=False,
                fp16=True,
            ),
        ),
    )

    # in_process=True runs vLLM in the main process so errors are visible
    backend = LocalBackend(in_process=True)
    from art.dev.openai_server import OpenAIServerConfig, ServerArgs
    await model.register(backend, _openai_client_config=OpenAIServerConfig(
        server_args=ServerArgs(port=8005),
    ))

    # ── Load Data ──
    all_scenarios = load_all_scenarios()

    # Filter to single category for focused training
    CATEGORY_FILTER = "Vague & Contextual"  # Set to "" to use all categories
    if CATEGORY_FILTER:
        all_scenarios = [s for s in all_scenarios if CATEGORY_FILTER in s.category]
        print(f"Filtered to '{CATEGORY_FILTER}': {len(all_scenarios)} scenarios")

    random.shuffle(all_scenarios)
    training_scenarios = all_scenarios

    print(f"Total scenarios: {len(all_scenarios)} (all used for training, no val split)")

    # ── Training Config ──
    training_config = {
        "groups_per_step": 1,
        "num_epochs": 5,
        "rollouts_per_group": 8,  # More rollouts = lower skip rate with binary rewards
        "learning_rate": 5e-6,
        "max_steps": 0,  # 0 = no limit (run all epochs)
    }

    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    # ── Diagnostic Log ──
    diagnostic_log: list[dict] = []
    from collections import Counter

    # ── Training Loop ──
    for batch in training_iterator:
        step_timer = StepTimer()
        step_timer.start("step_total")

        torch.cuda.reset_peak_memory_stats()

        print(
            f"\n{'='*60}\n"
            f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}\n"
            f"Batch contains {len(batch.items)} scenarios\n"
            f"{'='*60}"
        )

        gpu_before_rollouts = gpu_snapshot("before rollouts")

        # ── Rollouts ──
        step_timer.start("rollout_generation_s")

        train_groups = []
        for scenario in batch.items:
            train_groups.append(
                art.TrajectoryGroup(
                    (
                        rollout(
                            model,
                            CalendarRolloutInput(step=batch.step, scenario=scenario),
                        )
                        for _ in range(training_config["rollouts_per_group"])
                    )
                )
            )

        finished_train_groups = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )
        step_timer.stop()

        gpu_after_rollouts = gpu_snapshot("after rollouts")

        # ── Collect per-group and per-trajectory stats ──
        all_rewards = []
        group_details = []
        all_traj_metrics = []
        skip_count = 0

        for group in finished_train_groups:
            group_rewards = [t.reward for t in group.trajectories]
            group_verdicts = [
                t.metrics.get("verdict", -1) for t in group.trajectories
            ]
            scenario_id = group.trajectories[0].metadata.get("scenario_id", "?") if group.trajectories else "?"
            category = group.trajectories[0].metadata.get("category", "?") if group.trajectories else "?"

            # GRPO skip: all rewards identical → zero gradient
            skipped = len(set(group_rewards)) <= 1
            if skipped:
                skip_count += 1

            group_details.append({
                "scenario_id": scenario_id,
                "category": category,
                "rewards": group_rewards,
                "skipped": skipped,
                "verdicts": group_verdicts,
            })

            all_rewards.extend(group_rewards)
            for t in group.trajectories:
                all_traj_metrics.append(t.metrics)

        correct_count = sum(1 for r in all_rewards if r == 1.0)
        total_count = len(all_rewards)
        mean_reward = sum(all_rewards) / total_count if total_count else 0.0
        std_reward = (sum((r - mean_reward) ** 2 for r in all_rewards) / total_count) ** 0.5 if total_count else 0.0

        # Trajectory-level stats
        mean_turns = sum(m.get("num_turns", 0) for m in all_traj_metrics) / len(all_traj_metrics) if all_traj_metrics else 0
        no_answer_count = sum(1 for m in all_traj_metrics if m.get("no_final_answer", 0) > 0)
        tool_call_errors = sum(1 for m in all_traj_metrics if m.get("had_error", 0) > 0)
        mean_tool_calls = sum(m.get("num_tool_calls", 0) for m in all_traj_metrics) / len(all_traj_metrics) if all_traj_metrics else 0
        mean_judge_latency = sum(m.get("judge_latency_s", 0) for m in all_traj_metrics) / len(all_traj_metrics) if all_traj_metrics else 0
        total_prompt_tok = sum(m.get("prompt_tokens", 0) for m in all_traj_metrics)
        total_completion_tok = sum(m.get("completion_tokens", 0) for m in all_traj_metrics)

        # Tool name distribution (stored in metadata, not metrics)
        tool_counter: Counter = Counter()
        for group in finished_train_groups:
            for t in group.trajectories:
                names_str = t.metadata.get("tool_names", "")
                if isinstance(names_str, str) and names_str:
                    for n in names_str.split(","):
                        tool_counter[n.strip()] += 1

        # Compute inference tokens/sec from rollout time
        rollout_time = step_timer.records.get("rollout_generation_s", 1.0)
        inference_tps = round((total_prompt_tok + total_completion_tok) / rollout_time, 1) if rollout_time > 0 else 0

        print(f"\n  [STEP {batch.step} SUMMARY] reward={mean_reward:.3f}±{std_reward:.3f} "
              f"acc={correct_count}/{total_count} ({correct_count/total_count*100:.1f}%) "
              f"skip={skip_count}/{len(finished_train_groups)} ({skip_count/len(finished_train_groups)*100:.0f}%) "
              f"errors={tool_call_errors} no_answer={no_answer_count} "
              f"tokens={total_prompt_tok+total_completion_tok} tps={inference_tps}")

        # ── Checkpoint delete + Train ──
        step_timer.start("checkpoint_delete_s")
        await model.delete_checkpoints()
        step_timer.stop()

        gpu_before_train = gpu_snapshot("before train (pre-gc)")
        gc.collect()
        torch.cuda.empty_cache()
        gpu_snapshot("before train (post-gc+empty_cache)")

        step_timer.start("train_step_s")
        await model.train(
            finished_train_groups,
            config=art.TrainConfig(
                learning_rate=training_config["learning_rate"],
                beta=0.0,
            ),
            _config=dev.TrainConfig(logprob_calculation_chunk_size=128),
        )
        step_timer.stop()

        gpu_after_train = gpu_snapshot("after train")
        peak_allocated = round(torch.cuda.max_memory_allocated() / 1024**3, 3) if torch.cuda.is_available() else 0

        # Stop total timer
        step_timer.start("_dummy")
        step_timer.stop()
        timing = step_timer.get()
        # Compute step_total from all phases
        timing["step_total_s"] = round(sum(v for k, v in timing.items() if k != "_dummy"), 2)
        timing.pop("_dummy", None)
        timing.pop("step_total", None)

        # ── Build step record ──
        step_record = {
            "step": batch.step,
            "epoch": batch.epoch,
            "timestamp": datetime.now().isoformat(),
            "timing": timing,
            "gpu": {
                "before_rollouts": gpu_before_rollouts,
                "after_rollouts": gpu_after_rollouts,
                "before_train": gpu_before_train,
                "after_train": gpu_after_train,
                "peak_allocated_gb": peak_allocated,
            },
            "rewards": {
                "mean": round(mean_reward, 4),
                "std": round(std_reward, 4),
                "correct_count": correct_count,
                "total_count": total_count,
                "accuracy": round(correct_count / total_count, 4) if total_count else 0,
                "skip_count": skip_count,
                "skip_rate": round(skip_count / len(finished_train_groups), 4) if finished_train_groups else 0,
            },
            "groups": group_details,
            "trajectory_stats": {
                "mean_turns": round(mean_turns, 2),
                "no_answer_count": no_answer_count,
                "tool_call_errors": tool_call_errors,
                "mean_tool_calls": round(mean_tool_calls, 2),
                "tool_name_distribution": dict(tool_counter),
                "mean_judge_latency_s": round(mean_judge_latency, 2),
                "total_prompt_tokens": int(total_prompt_tok),
                "total_completion_tokens": int(total_completion_tok),
                "inference_tokens_per_sec": inference_tps,
            },
        }
        diagnostic_log.append(step_record)

        print(f"\n  Completed training step {batch.step} in {timing.get('step_total_s', '?')}s")

        if training_config["max_steps"] > 0 and batch.step >= training_config["max_steps"]:
            break

    # ── Save diagnostic log ──
    diag_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rl_diagnostic.json")
    with open(diag_path, "w") as f:
        json.dump(diagnostic_log, f, indent=2, default=str)
    print(f"\nDiagnostic log saved to {diag_path}")
    print("Training complete.")


if __name__ == "__main__":
    asyncio.run(main())
    # ART/vLLM daemon threads don't shut down cleanly — force exit to free GPU
    os._exit(0)
