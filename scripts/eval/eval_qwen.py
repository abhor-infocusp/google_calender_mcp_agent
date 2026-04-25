#!/usr/bin/env python3
"""Evaluate a Qwen (or any OpenAI-compatible) model on calendar tasks.

Uses the same calendar environment, tools, and Gemini eval judge as
run_trajectory.py, but drives the agent via an OpenAI-compatible API
(e.g. vLLM serving Qwen3-4B).

Usage:
    # Base model
    python eval_qwen.py 5 --query-index 4 --model Qwen/Qwen3-4B

    # Trained model with return_final_answer
    python eval_qwen.py 5 --query-index 4 --model trained --with-final-answer

    # All queries for a calendar
    python eval_qwen.py 5 --model Qwen/Qwen3-4B

Requires a vLLM server running:
    vllm serve Qwen/Qwen3-4B --enable-auto-tool-choice \\
        --tool-call-parser hermes --max-model-len 3072
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from openai import OpenAI

from calendar_agent.environment import CalendarEnvironment
from calendar_agent.core import (
    DAY_NAMES,
    C,
    compute_fallback_now,
    diff_snapshots,
    dispatch_tool_call,
    filter_by_days,
    format_day_state,
    get_query_now,
    print_agent_text,
    print_separator,
    print_tool_call,
    print_tool_result,
    snapshot_events,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, evaluate_trajectory
from calendar_agent.paths import SFT_DATA_DIR, DATA_DIR, RL_DATA_DIR, CREDENTIALS_PATH
from calendar_agent.core import format_tool_result
from calendar_agent.tools import get_openai_tools_minimal, RETURN_FINAL_ANSWER_TOOL_MINIMAL

import vertexai
from vertexai.generative_models import GenerativeModel
from google.oauth2.credentials import Credentials as OAuth2Credentials

# ── Data Loading (supports both data/ and sft_data/) ──────

SFT_DATA_DIR = str(SFT_DATA_DIR)
DATA_DIR = str(DATA_DIR)
RL_DATA_DIR = str(RL_DATA_DIR)


def load_calendar_and_queries(index: int, use_sft_data: bool = False, use_rl_data: bool = False):
    """Load calendar events and queries for the given index."""
    base = RL_DATA_DIR if use_rl_data else (SFT_DATA_DIR if use_sft_data else DATA_DIR)
    cal_path = os.path.join(base, "json_calender", f"{index}.txt")
    query_path = os.path.join(base, "queries", f"{index}.txt")

    if not os.path.exists(cal_path):
        sys.exit(f"Calendar file not found: {cal_path}")
    if not os.path.exists(query_path):
        sys.exit(f"Query file not found: {query_path}")

    events = CalendarEnvironment.load_json_calendar(cal_path)

    with open(query_path) as f:
        queries = json.load(f)

    fallback_now = compute_fallback_now(cal_path)

    return events, queries, fallback_now

OPENAI_TOOLS = get_openai_tools_minimal()


# ── OpenAI Agent Loop ────────────────────────────────────────


def run_query_openai(
    client: OpenAI,
    model_name: str,
    tools: list[dict],
    system_prompt: str,
    env: CalendarEnvironment,
    query: str,
    max_turns: int,
) -> list[dict]:
    """Run a single query using an OpenAI-compatible API.

    Returns the same trajectory format as run_trajectory.run_query().
    """
    trajectory = []
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    print(f"\n  {C.BLUE}[USER]       {query}{C.RESET}")
    trajectory.append({"role": "user", "content": query})

    for turn in range(1, max_turns + 1):
        print(f"\n  -- Turn {turn} --")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                temperature=0.7,
            )
        except Exception as e:
            print(f"  [ERROR]      Model call failed: {e}")
            trajectory.append({"role": "error", "content": str(e)})
            break

        choice = response.choices[0]
        msg = choice.message

        # Build the assistant message dict for the conversation history
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # Show any text the model produced
        if msg.content:
            has_tool_calls = bool(msg.tool_calls)
            print_agent_text(msg.content, is_final=not has_tool_calls)
            trajectory.append({"role": "assistant", "content": msg.content})

        # If no tool calls, the agent is done
        if not msg.tool_calls:
            break

        # Execute each tool call
        hit_final_answer = False
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            # Handle return_final_answer as terminal
            if tool_name == "return_final_answer":
                answer = args.get("answer", "")
                print_agent_text(answer, is_final=True)
                trajectory.append({"role": "assistant", "content": answer})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": answer,
                    }
                )
                hit_final_answer = True
                break

            print_tool_call(tool_name, args)

            result = dispatch_tool_call(env, tool_name, args)
            result = format_tool_result(result)

            print_tool_result(result)
            trajectory.append(
                {
                    "role": "tool_call",
                    "name": tool_name,
                    "args": args,
                    "result": result,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result if isinstance(result, str) else json.dumps(result, default=str),
                }
            )

        if hit_final_answer:
            break
    else:
        print(f"\n  [MAX TURNS]  Reached limit of {max_turns} turns.")

    return trajectory


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a Qwen/OpenAI-compatible model on calendar tasks."
    )
    parser.add_argument(
        "calendar_index", type=int, help="Index of the calendar/query pair (0-49)."
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-4B",
        help="Model name as served by vLLM (default: Qwen/Qwen3-4B).",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible API base URL (default: http://localhost:8000/v1).",
    )
    parser.add_argument(
        "--api-key",
        default="token-abc123",
        help="API key for the OpenAI-compatible server (default: token-abc123).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max tool-use turns per query (default: 10).",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Run only a specific query by index (0-based).",
    )
    parser.add_argument(
        "--with-final-answer",
        action="store_true",
        help="Include return_final_answer tool (for trained models).",
    )
    parser.add_argument(
        "--sft-data",
        action="store_true",
        help="Load calendars/queries from sft_data/ instead of data/.",
    )
    parser.add_argument(
        "--rl-data",
        action="store_true",
        help="Load calendars/queries from rl_data/ instead of data/.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save full trajectories to this JSON file.",
    )
    args = parser.parse_args()

    # Load data
    events, queries, fallback_now = load_calendar_and_queries(
        args.calendar_index, use_sft_data=args.sft_data, use_rl_data=args.rl_data
    )

    # OpenAI client for the agent model
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # Match SFT training distribution: /no_think + calendar assistant preamble.
    tools = list(OPENAI_TOOLS)
    system_prompt = "/no_think\nYou are a calendar assistant. Use the provided tools to manage events. Call get_current_time first to know the current date."
    if args.with_final_answer:
        tools.append(RETURN_FINAL_ANSWER_TOOL_MINIMAL)

    # Init Vertex AI for the Gemini eval judge
    _gcp_credentials = None
    _creds_path = str(CREDENTIALS_PATH)
    if os.path.exists(_creds_path):
        with open(_creds_path) as _f:
            _cred_data = json.load(_f)
        _gcp_credentials = OAuth2Credentials(
            token=None,
            refresh_token=_cred_data["refresh_token"],
            client_id=_cred_data["client_id"],
            client_secret=_cred_data["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
        )
    vertexai.init(
        project=os.environ.get("GCP_PROJECT", "internal-ml-exp"),
        location=os.environ.get("GCP_LOCATION", "us-central1"),
        credentials=_gcp_credentials,
    )
    eval_model = GenerativeModel(
        "gemini-2.0-flash-001",
        system_instruction=[EVAL_SYSTEM_PROMPT],
    )

    # Header
    print_separator()
    print(
        f"  Qwen Eval | Calendar {args.calendar_index} | Model: {args.model}"
    )
    print(f"  Backend   : {args.base_url}")
    print_separator()
    print(f"  Events loaded : {len(events)}")
    print(f"  Queries       : {len(queries)}")
    print(f"  Fallback now  : {fallback_now}")

    # Select queries to run
    if args.query_index is not None:
        if args.query_index >= len(queries):
            sys.exit(
                f"Query index {args.query_index} out of range (0-{len(queries)-1})."
            )
        selected = [(args.query_index, queries[args.query_index])]
    else:
        selected = list(enumerate(queries))

    all_trajectories = []

    for qi, q in selected:
        now = get_query_now(q, fallback_now)
        env = CalendarEnvironment()
        env.initialize(events=events, now=now)

        category = q.get("category", "N/A")
        complexity = q.get("complexity", "N/A")
        query_text = q["query"]
        expected = q.get("expected_behavior", "")

        print()
        print_separator("-")
        print(
            f"  QUERY {qi + 1}/{len(queries)} | {category} | Complexity: {complexity}"
        )
        print(f"  Simulated now : {now}")
        if expected:
            print(f"  Expected: {expected}")
        print_separator("-")

        addressed_days = q.get("addressed_days", [])
        display_days = addressed_days if addressed_days else DAY_NAMES

        before = snapshot_events(env)
        before_days = filter_by_days(before, display_days)

        label = ", ".join(addressed_days) if addressed_days else "all days"
        print(f"\n  [BEFORE] Calendar state for: {label}")
        for line in format_day_state(before_days):
            print(f"  {line}")

        trajectory = run_query_openai(
            client, args.model, tools, system_prompt, env, query_text, args.max_turns
        )

        after = snapshot_events(env)
        after_days = filter_by_days(after, display_days)

        print(f"\n  [AFTER] Calendar state for: {label}")
        for line in format_day_state(after_days):
            print(f"  {line}")

        changes = diff_snapshots(before, after)
        if changes:
            print(f"\n  [CHANGES]")
            for line in changes:
                print(f"  {line}")
        else:
            print(f"\n  [CHANGES]  (none)")

        # Evaluation via Gemini judge
        final_output = next(
            (
                step["content"]
                for step in reversed(trajectory)
                if step["role"] == "assistant"
            ),
            "",
        )
        print()
        print_separator("·")
        verdict, _reasoning = evaluate_trajectory(
            eval_model,
            query_text,
            final_output,
            expected,
            before_days,
            after_days,
        )
        verdict_color = (
            C.GREEN
            if verdict == "Correct"
            else C.RED if verdict == "Incorrect" else C.YELLOW
        )
        print(f"\n  {verdict_color}[EVAL RESULT] {verdict}{C.RESET}")

        all_trajectories.append(
            {
                "query_index": qi,
                "category": category,
                "complexity": complexity,
                "query": query_text,
                "simulated_now": now,
                "expected_behavior": expected,
                "addressed_days": addressed_days,
                "calendar_before": before_days,
                "calendar_after": after_days,
                "state_changes": changes,
                "trajectory": trajectory,
                "eval_verdict": verdict,
            }
        )

    # Summary
    print()
    print_separator()
    print(f"  Done. Ran {len(selected)} queries.")
    verdicts = [t["eval_verdict"] for t in all_trajectories]
    correct = verdicts.count("Correct")
    incorrect = verdicts.count("Incorrect")
    unsure = verdicts.count("Unsure")
    print(
        f"  Eval results  : {C.GREEN}{correct} Correct{C.RESET}  "
        f"{C.RED}{incorrect} Incorrect{C.RESET}  "
        f"{C.YELLOW}{unsure} Unsure{C.RESET}"
    )
    print_separator()

    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_trajectories, f, indent=2, default=str)
        print(f"  Trajectories saved to {args.save}")


if __name__ == "__main__":
    main()
