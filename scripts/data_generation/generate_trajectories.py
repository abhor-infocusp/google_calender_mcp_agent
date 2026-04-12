#!/usr/bin/env python3
"""Generate SFT trajectories using gemini-2.5-pro with tuned prompt.

Runs each query through the agent loop (up to 3 attempts), evaluates
correctness, and saves solved (Correct) trajectories to sft_data/trajectories/.

Usage:
    PYTHONPATH=src python scripts/data_generation/generate_trajectories.py
    PYTHONPATH=src python scripts/data_generation/generate_trajectories.py --prompt-file prompts/v11_reason_act.txt
"""

import argparse
import json
import glob
import os
import sys
import uuid
import warnings
import time

warnings.filterwarnings("ignore")

import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
import google.auth.transport.requests
from vertexai.generative_models import GenerativeModel, Part

from calendar_agent.environment import CalendarEnvironment
from calendar_agent.core import (
    CALENDAR_TOOL,
    SYSTEM_PROMPT,
    compute_fallback_now,
    dispatch_tool_call,
    snapshot_events,
    filter_by_days,
    get_query_now,
    DAY_NAMES,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, evaluate_trajectory
from calendar_agent.paths import SFT_DATA_DIR as _SFT_DATA_DIR, SFT_JSON_CALENDAR_DIR, SFT_QUERY_DIR, CREDENTIALS_PATH
from calendar_agent.core import format_tool_result

sys.stdout.reconfigure(line_buffering=True)

DEFAULT_MODEL = "gemini-2.0-flash-001"

# COST GUARD: gemini-2.5-pro used 20x monthly budget in 1 day during Phase 3.
# Block pro models unless explicitly overridden with --allow-pro.
_BLOCKED_MODELS = {"gemini-2.5-pro", "gemini-1.5-pro", "gemini-2.5-pro-preview-05-06"}
DEFAULT_PROMPT = os.path.join(os.path.dirname(__file__), "../../prompts/v11_reason_act.txt")
MAX_TURNS = 10
MAX_ATTEMPTS = 3

SFT_DATA_DIR = str(_SFT_DATA_DIR)
JSON_CALENDAR_DIR = str(SFT_JSON_CALENDAR_DIR)
QUERY_DIR = str(SFT_QUERY_DIR)
TRAJ_DIR = os.path.join(SFT_DATA_DIR, "trajectories")


def run_single_trajectory(model, eval_model, cal_path, query_dict, max_turns=MAX_TURNS):
    """Run a single query, evaluate it, and return (verdict, trajectory_data)."""
    events = CalendarEnvironment.load_json_calendar(cal_path)

    fallback_now = compute_fallback_now(cal_path)
    now = get_query_now(query_dict, fallback_now)
    env = CalendarEnvironment()
    env.initialize(events=events, now=now)

    addressed_days = query_dict.get("addressed_days", [])
    display_days = addressed_days if addressed_days else DAY_NAMES

    before = snapshot_events(env)
    before_days = filter_by_days(before, display_days)

    # Run trajectory
    chat = model.start_chat()
    query_text = query_dict["query"]
    trajectory = []
    trajectory.append({"role": "user", "content": query_text})

    try:
        response = chat.send_message(query_text)
    except Exception as e:
        return "Error", None

    for turn in range(1, max_turns + 1):
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
            trajectory.append({"role": "assistant", "content": "\n".join(text_parts)})

        if not function_calls:
            break

        response_parts = []
        for fc in function_calls:
            args = dict(fc.args)
            result = dispatch_tool_call(env, fc.name, args)
            result = format_tool_result(result)
            trajectory.append({
                "role": "tool_call",
                "name": fc.name,
                "args": args,
                "result": result,
            })
            # Vertex AI requires dict for function responses; wrap lists
            resp_for_vertex = result if isinstance(result, dict) else {"result": result}
            response_parts.append(
                Part.from_function_response(name=fc.name, response=resp_for_vertex)
            )

        try:
            response = chat.send_message(response_parts)
        except Exception as e:
            trajectory.append({"role": "error", "content": str(e)})
            break

    after = snapshot_events(env)
    after_days = filter_by_days(after, display_days)

    final_output = next(
        (step["content"] for step in reversed(trajectory) if step["role"] == "assistant"),
        "",
    )

    expected = query_dict.get("expected_behavior", "")
    verdict = evaluate_trajectory(
        eval_model, query_text, final_output, expected, before_days, after_days
    )

    traj_data = {
        "query": query_text,
        "category": query_dict.get("category", ""),
        "complexity": query_dict.get("complexity", ""),
        "expected_behavior": expected,
        "simulated_now": now,
        "addressed_days": addressed_days,
        "calendar_before": before_days,
        "calendar_after": after_days,
        "trajectory": trajectory,
        "eval_verdict": verdict,
    }

    return verdict, traj_data


def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Teacher model (default: {DEFAULT_MODEL})")
    parser.add_argument("--prompt-file", default=DEFAULT_PROMPT, help="System prompt file")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS, help="Max retry attempts per query (default: 3)")
    parser.add_argument("--allow-pro", action="store_true", help="Override pro model cost guard")
    args = parser.parse_args()

    if args.model in _BLOCKED_MODELS and not args.allow_pro:
        print(f"ERROR: {args.model} is blocked — used 20x monthly budget in 1 day.")
        print(f"  Use --allow-pro to override, or switch to gemini-2.0-flash-001.")
        sys.exit(1)

    os.makedirs(TRAJ_DIR, exist_ok=True)

    # Load system prompt
    prompt_path = os.path.abspath(args.prompt_file)
    with open(prompt_path) as f:
        system_prompt = f.read().strip()
    print(f"Prompt: {prompt_path} ({len(system_prompt)} chars)")

    # Init credentials
    with open(str(CREDENTIALS_PATH)) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None,
        refresh_token=cd["refresh_token"],
        client_id=cd["client_id"],
        client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
    refresh_req = google.auth.transport.requests.Request()

    def make_models():
        gen = GenerativeModel(args.model, tools=[CALENDAR_TOOL], system_instruction=[system_prompt])
        ev = GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])
        return gen, ev

    model, eval_model = make_models()

    # Collect all queries
    all_tasks = []
    for f in sorted(glob.glob(os.path.join(QUERY_DIR, "*.txt"))):
        cal_idx = os.path.basename(f).replace(".txt", "")
        cal_path = os.path.join(JSON_CALENDAR_DIR, f"{cal_idx}.txt")
        if not os.path.exists(cal_path):
            print(f"  Skipping cal {cal_idx}: no json_calender file")
            continue
        queries = json.load(open(f))
        for qi, q in enumerate(queries):
            all_tasks.append({
                "cal_idx": cal_idx,
                "query_idx": qi,
                "query": q,
                "cal_path": cal_path,
            })

    print(f"Total queries to process: {len(all_tasks)}")
    print(f"Model: {args.model}")
    print(f"Max attempts: {args.max_attempts}")
    print(f"Output: {TRAJ_DIR}/")
    print()

    correct_count = 0
    error_count = 0
    total_processed = 0

    # Process per calendar index for organized output
    cal_indices = sorted(set(t["cal_idx"] for t in all_tasks), key=lambda x: int(x))

    for cal_idx in cal_indices:
        cal_tasks = [t for t in all_tasks if t["cal_idx"] == cal_idx]
        solved_trajectories = []

        # Check if already processed
        out_path = os.path.join(TRAJ_DIR, f"{cal_idx}.json")
        if os.path.exists(out_path):
            existing = json.load(open(out_path))
            print(f"Cal {cal_idx}: already exists with {len(existing)} solved trajectories, skipping")
            correct_count += len(existing)
            total_processed += len(cal_tasks)
            continue

        print(f"Cal {cal_idx}: processing {len(cal_tasks)} queries...")

        for task in cal_tasks:
            qi = task["query_idx"]
            q = task["query"]
            complexity = q.get("complexity", "?")
            query_text = q["query"][:50]

            total_processed += 1
            print(f"  [{total_processed}/{len(all_tasks)}] q={qi} [{complexity}] {query_text}...", end=" ", flush=True)

            # Refresh token if needed
            if not creds.valid or creds.expired:
                creds.refresh(refresh_req)
                vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
                model, eval_model = make_models()
                print("[TOKEN REFRESHED] ", end="", flush=True)

            # Retry loop: up to max_attempts, stop on first Correct
            verdict = "Incorrect"
            traj_data = None
            for attempt in range(1, args.max_attempts + 1):
                try:
                    attempt_verdict, attempt_traj = run_single_trajectory(
                        model, eval_model, task["cal_path"], q
                    )
                except Exception as e:
                    attempt_verdict = "Error"
                    attempt_traj = None

                if attempt_verdict == "Correct":
                    verdict = "Correct"
                    traj_data = attempt_traj
                    if attempt > 1:
                        print(f"Correct (attempt {attempt}, saved)", flush=True)
                    else:
                        print("Correct (saved)", flush=True)
                    break
                else:
                    if attempt < args.max_attempts:
                        print(f"retry {attempt}...", end=" ", flush=True)
                        time.sleep(0.3)
                    else:
                        print(f"Incorrect ({args.max_attempts}/{args.max_attempts})", flush=True)

            if verdict == "Correct":
                correct_count += 1
                solved_trajectories.append(traj_data)
            elif verdict == "Error":
                error_count += 1

            time.sleep(0.5)

        # Save solved trajectories for this calendar
        if solved_trajectories:
            with open(out_path, "w") as f:
                json.dump(solved_trajectories, f, indent=2, default=str)
            print(f"  -> Saved {len(solved_trajectories)} solved trajectories to {out_path}")
        else:
            print(f"  -> No solved trajectories for cal {cal_idx}")

        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total queries:      {len(all_tasks)}")
    print(f"  Correct (saved):    {correct_count}")
    print(f"  Errors:             {error_count}")
    print(f"  Incorrect/skipped:  {len(all_tasks) - correct_count - error_count}")
    print(f"  Solve rate:         {correct_count/len(all_tasks):.1%}")
    print(f"  Output directory:   {TRAJ_DIR}/")

    # Count total saved files
    saved_files = glob.glob(os.path.join(TRAJ_DIR, "*.json"))
    total_saved = sum(len(json.load(open(f))) for f in saved_files)
    print(f"  Total saved trajectories: {total_saved}")


if __name__ == "__main__":
    main()
