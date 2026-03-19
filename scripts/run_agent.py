#!/usr/bin/env python3
"""Run an agent trajectory against a loaded calendar.

Loads a calendar from data/json_calender/<index>.txt and the corresponding
queries from data/queries/<index>.txt. For each query, runs an agentic loop
where the model uses calendar tools to fulfill the request.

Usage:
    python scripts/run_agent.py 0
    python scripts/run_agent.py 0 --query-index 3
    python scripts/run_agent.py 0 --model gemini-2.0-flash-001 --max-turns 15
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from calendar_agent.core import (
    C,
    CALENDAR_TOOL,
    DAY_NAMES,
    SYSTEM_PROMPT,
    dispatch_tool_call,
    filter_by_days,
    format_day_state,
    get_query_now,
    load_calendar_and_queries,
    print_agent_text,
    print_separator,
    print_tool_call,
    print_tool_result,
    snapshot_events,
    diff_snapshots,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, evaluate_trajectory
from calendar_agent.environment import CalendarEnvironment
from calendar_agent.paths import CREDENTIALS_PATH
from calendar_agent.tools import serialize_tool_result


# ── Core Agent Loop ──────────────────────────────────────────


def run_query(
    model, env: CalendarEnvironment, query: str, max_turns: int
) -> list[dict]:
    """Run a single query through the agentic tool-use loop."""
    trajectory = []
    chat = model.start_chat()

    print(f"\n  {C.BLUE}[USER]       {query}{C.RESET}")
    trajectory.append({"role": "user", "content": query})

    try:
        response = chat.send_message(query)
    except Exception as e:
        print(f"  [ERROR]      Model call failed: {e}")
        trajectory.append({"role": "error", "content": str(e)})
        return trajectory

    for turn in range(1, max_turns + 1):
        print(f"\n  -- Turn {turn} --")

        function_calls = []
        text_parts = []
        for part in response.candidates[0].content.parts:
            try:
                if part.function_call.name:
                    function_calls.append(part.function_call)
                    continue
            except AttributeError:
                pass
            try:
                if part.text:
                    text_parts.append(part.text)
            except AttributeError:
                pass

        if text_parts:
            combined = "\n".join(text_parts)
            print_agent_text(combined, is_final=not function_calls)
            trajectory.append({"role": "assistant", "content": combined})

        if not function_calls:
            break

        response_parts = []
        for fc in function_calls:
            args = dict(fc.args)
            print_tool_call(fc.name, args)

            result = dispatch_tool_call(env, fc.name, args)
            if result is None:
                result = {"status": "ok"}
            result = serialize_tool_result(result)

            print_tool_result(result)
            trajectory.append(
                {
                    "role": "tool_call",
                    "name": fc.name,
                    "args": args,
                    "result": result,
                }
            )

            response_parts.append(
                Part.from_function_response(name=fc.name, response=result)
            )

        try:
            response = chat.send_message(response_parts)
        except Exception as e:
            print(f"  [ERROR]      Model call failed: {e}")
            trajectory.append({"role": "error", "content": str(e)})
            break
    else:
        print(f"\n  [MAX TURNS]  Reached limit of {max_turns} turns.")

    return trajectory


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run agent trajectories on calendar data."
    )
    parser.add_argument(
        "calendar_index", type=int, help="Index of the calendar/query pair (0-49)."
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT", "internal-ml-exp"),
        help="GCP project ID (default: $GCP_PROJECT or 'internal-ml-exp').",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("GCP_LOCATION", "us-central1"),
        help="GCP location (default: $GCP_LOCATION or 'us-central1').",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash-001",
        help="Gemini model name (default: gemini-2.0-flash-001).",
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
        "--save",
        type=str,
        default=None,
        help="Save full trajectories to this JSON file.",
    )
    args = parser.parse_args()

    events, queries, fallback_now = load_calendar_and_queries(args.calendar_index)

    _gcp_credentials = None
    if CREDENTIALS_PATH.exists():
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
    vertexai.init(
        project=args.project, location=args.location, credentials=_gcp_credentials
    )
    model = GenerativeModel(
        args.model,
        tools=[CALENDAR_TOOL],
        system_instruction=[SYSTEM_PROMPT],
    )
    eval_model = GenerativeModel(
        args.model,
        system_instruction=[EVAL_SYSTEM_PROMPT],
    )

    print_separator()
    print(f"  Trajectory Runner | Calendar {args.calendar_index} | Model: {args.model}")
    print_separator()
    print(f"  Events loaded : {len(events)}")
    print(f"  Queries       : {len(queries)}")
    print(f"  Fallback now  : {fallback_now}")

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

        label = ', '.join(addressed_days) if addressed_days else "all days"
        print(f"\n  [BEFORE] Calendar state for: {label}")
        for line in format_day_state(before_days):
            print(f"  {line}")

        trajectory = run_query(model, env, query_text, args.max_turns)

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
        verdict = evaluate_trajectory(
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
