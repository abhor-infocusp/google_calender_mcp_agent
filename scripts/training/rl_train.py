import asyncio
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
    SYSTEM_PROMPT,
    compute_fallback_now,
    dispatch_tool_call,
    filter_by_days,
    snapshot_events,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, format_day_state_text
from calendar_agent.paths import RL_DATA_DIR as _RL_DATA_DIR, RL_JSON_CALENDAR_DIR, RL_QUERY_DIR, CREDENTIALS_PATH
from calendar_agent.tools import get_openai_tools

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


OPENAI_TOOLS = get_openai_tools(include_final_answer=True)

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

SYSTEM_PROMPT_TEMPLATE = (
    SYSTEM_PROMPT
    + "- When you are done, call return_final_answer with a summary of what you did.\n\n"
    + "You may take up to {max_turns} turns to complete the task.\n"
)


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

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(max_turns=MAX_TURNS)

    traj.messages_and_choices = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": scenario.query},
    ]

    traj.tools = OPENAI_TOOLS

    client = AsyncOpenAI(
        base_url=model.inference_base_url,
        api_key=model.inference_api_key,
    )

    final_answer_text = None

    # ── Agent loop ──
    for _ in range(MAX_TURNS):
        try:
            response = await client.chat.completions.create(
                model=model.get_inference_name(),
                temperature=1,
                messages=traj.messages(),
                tools=traj.tools,
            )
        except Exception as e:
            print(f"[ROLLOUT ERROR] LLM call failed: {e}")
            traceback.print_exc()
            break

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

                # Dispatch to CalendarEnvironment via run_trajectory helper
                result = dispatch_tool_call(env, tool_name, tool_args)
                result_str = json.dumps(result, default=str)
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

    if final_answer_text is not None:
        # Run Gemini judge for full correctness
        after_snap = snapshot_events(env)
        before_days = filter_by_days(before_snap, scenario.addressed_days)
        after_days = filter_by_days(after_snap, scenario.addressed_days)

        verdict = await evaluate_trajectory(
            query=scenario.query,
            final_output=final_answer_text,
            expected=scenario.expected_behavior,
            before_days=before_days,
            after_days=after_days,
        )

        if verdict == "Correct":
            reward = 1.0

    traj.reward = reward
    traj.metrics["correct"] = 1.0 if verdict == "Correct" else 0.0
    traj.metrics["verdict"] = {"Correct": 1, "Incorrect": 0}.get(verdict, -1)
    traj.metrics["shaped_reward"] = reward
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


# ── Main ───────────────────────────────────────────────────


async def main():
    # ── Model & Backend ──
    model = art.TrainableModel(
        name="calendar-agent-001",
        project="calendar-agent",
        # Must be the MERGED fp16 model from serialized SFT training.
        # Create with: merge_lora.py or Unsloth save_pretrained_merged()
        # from the best SFT checkpoint (e.g. sft_output/checkpoint-144).
        base_model="sft_output/merged_instruct",
        _internal_config=dev.InternalModelConfig(
            init_args=dev.InitArgs(
                load_in_4bit=True,
                use_bitsandbytes=False,
                max_lora_rank=32,
            ),
            engine_args=dev.EngineArgs(
                max_model_len=2048,
                max_num_batched_tokens=2048,
                max_num_seqs=4,
                gpu_memory_utilization=0.55,
                enforce_eager=True,
                swap_space=2,
                enable_sleep_mode=False,
            ),
            trainer_args=dev.TrainerArgs(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
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
    random.shuffle(all_scenarios)

    # 90/10 train/val split
    split_point = int(len(all_scenarios) * 0.9)
    training_scenarios = all_scenarios[:split_point]
    validation_scenarios = all_scenarios[split_point:]

    print(f"Total scenarios: {len(all_scenarios)}")
    print(
        f"Training: {len(training_scenarios)}, Validation: {len(validation_scenarios)}"
    )

    # ── Training Config ──
    training_config = {
        "groups_per_step": 8,
        "num_epochs": 3,
        "rollouts_per_group": 4,  # Need >=4 for binary rewards at ~10% correct rate
        "learning_rate": 5e-6,
        "max_steps": 500,
        "validation_step_interval": 15,
    }

    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    # ── Training Loop ──
    for batch in training_iterator:
        print(
            f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
        )
        print(f"Batch contains {len(batch.items)} scenarios")
        print_gpu_usage("before rollouts")

        # Create trajectory groups for this batch
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

        # Gather all trajectory groups
        finished_train_groups = await art.gather_trajectory_groups(
            train_groups,
            pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )
        print_gpu_usage("after rollouts")

        # Validation
        if batch.step % training_config["validation_step_interval"] == 0:
            print("Running validation at step", batch.step)
            validation_groups = []
            val_sample = random.sample(
                validation_scenarios, min(5, len(validation_scenarios))
            )
            for scenario in val_sample:
                validation_groups.append(
                    art.TrajectoryGroup(
                        [
                            rollout(
                                model,
                                CalendarRolloutInput(
                                    step=batch.step, scenario=scenario
                                ),
                            )
                        ]
                    )
                )

            finished_validation_groups = await art.gather_trajectory_groups(
                validation_groups,
                pbar_desc="validation",
                max_exceptions=len(validation_scenarios),
            )

            await model.log(
                finished_validation_groups,
                split="val",
            )
            print_gpu_usage("after validation")

        await model.delete_checkpoints()
        await model.train(
            finished_train_groups,
            config=art.TrainConfig(
                learning_rate=training_config["learning_rate"],
                beta=0.0,
            ),
            _config=dev.TrainConfig(logprob_calculation_chunk_size=128),
        )
        print_gpu_usage("after train")

        print(f"Completed training step {batch.step}")

        if batch.step >= training_config["max_steps"]:
            break

    print("Training complete.")


if __name__ == "__main__":
    asyncio.run(main())
