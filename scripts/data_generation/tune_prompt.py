#!/usr/bin/env python3
"""Fast prompt tuning for Gemini teacher models.

Runs a single model on filtered queries with a custom system prompt.
Appends results to sft_data/prompt_tuning_log.jsonl for comparison.

Usage:
    PYTHONPATH=src python scripts/data_generation/tune_prompt.py \
        --categories "Human Chaos" --prompt-file prompts/v2_chaos.txt --prompt-id v2_chaos

    # All categories, default prompt:
    PYTHONPATH=src python scripts/data_generation/tune_prompt.py --prompt-id baseline

    # Specific calendars:
    PYTHONPATH=src python scripts/data_generation/tune_prompt.py --calendars 0 2 3
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime

warnings.filterwarnings("ignore")

import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
import google.auth.transport.requests
from vertexai.generative_models import GenerativeModel, Part

from calendar_agent.core import CALENDAR_TOOL, SYSTEM_PROMPT
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT
from calendar_agent.paths import SFT_DATA_DIR as _SFT_DATA_DIR, CREDENTIALS_PATH

sys.stdout.reconfigure(line_buffering=True)

SFT_DATA_DIR = str(_SFT_DATA_DIR)
JSON_CALENDAR_DIR = os.path.join(SFT_DATA_DIR, "json_calender")
QUERY_DIR = os.path.join(SFT_DATA_DIR, "queries")
LOG_PATH = os.path.join(SFT_DATA_DIR, "prompt_tuning_log.jsonl")

DEFAULT_CALENDARS = [0, 2, 3, 4, 5]


def run_single(model, eval_model, cal_path, query_dict, max_turns=10):
    """Run a single trajectory and return verdict. Identical to compare_teachers.py."""
    from calendar_agent.environment import CalendarEnvironment
    from calendar_agent.core import (
        compute_fallback_now, dispatch_tool_call, format_tool_result,
        snapshot_events, filter_by_days, get_query_now, DAY_NAMES,
    )
    from calendar_agent.evaluation import format_day_state_text

    events = CalendarEnvironment.load_json_calendar(cal_path)
    fallback_now = compute_fallback_now(cal_path)
    now = get_query_now(query_dict, fallback_now)
    env = CalendarEnvironment()
    env.initialize(events=events, now=now)

    addressed_days = query_dict.get("addressed_days", [])
    display_days = addressed_days if addressed_days else DAY_NAMES
    before = snapshot_events(env)
    before_days = filter_by_days(before, display_days)

    chat = model.start_chat()
    query_text = query_dict["query"]
    trajectory = [{"role": "user", "content": query_text}]

    try:
        response = chat.send_message(query_text)
    except Exception as e:
        print(f"\n    [INIT ERROR] {type(e).__name__}: {e}", flush=True)
        return "Error", None

    for turn in range(1, max_turns + 1):
        # Log raw response for debugging
        finish_reason = None
        try:
            finish_reason = response.candidates[0].finish_reason
        except Exception:
            pass

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
            if finish_reason and str(finish_reason) != "1" and str(finish_reason) != "FinishReason.STOP":
                trajectory.append({"role": "assistant", "content": f"[RESPONSE CUT OFF: finish_reason={finish_reason}, turn={turn}]"})
            break

        response_parts = []
        for fc in function_calls:
            args = dict(fc.args)
            result = dispatch_tool_call(env, fc.name, args)
            result = format_tool_result(result)
            trajectory.append({"role": "tool_call", "name": fc.name, "args": args, "result": result})
            resp_for_vertex = result if isinstance(result, dict) else {"result": result}
            response_parts.append(Part.from_function_response(name=fc.name, response=resp_for_vertex))

        try:
            response = chat.send_message(response_parts)
        except Exception as e:
            trajectory.append({"role": "assistant", "content": f"[SEND ERROR: {type(e).__name__}: {e}]"})
            break

    after = snapshot_events(env)
    after_days = filter_by_days(after, display_days)
    final_output = next(
        (s["content"] for s in reversed(trajectory) if s["role"] == "assistant"), ""
    )
    expected = query_dict.get("expected_behavior", "")

    # Inline eval to capture full judge reasoning
    before_text = format_day_state_text(before_days)
    after_text = format_day_state_text(after_days)
    eval_prompt = f"""\
Query: {query_text}

Response: {final_output if final_output else "(no response)"}

Expected: {expected if expected else "(not specified)"}

Before:
{before_text}

After:
{after_text}

Was the task completed correctly? First explain your reasoning in 2-3 sentences. Then on the very last line output exactly one word: Correct or Incorrect."""

    judge_raw = ""
    verdict = "Incorrect"
    try:
        import signal
        def _timeout_handler(signum, frame):
            raise TimeoutError("Gemini eval timed out")
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(30)
        try:
            response = eval_model.generate_content(eval_prompt)
            judge_raw = response.text.strip()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        lines = [l.strip() for l in judge_raw.splitlines() if l.strip()]
        for line in reversed(lines):
            if line.lower() == "incorrect":
                verdict = "Incorrect"; break
            if line.lower() == "correct":
                verdict = "Correct"; break
        else:
            for line in reversed(lines):
                if "incorrect" in line.lower():
                    verdict = "Incorrect"; break
                if "correct" in line.lower():
                    verdict = "Correct"; break
    except Exception as e:
        judge_raw = f"[EVAL ERROR] {e}"

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
        "judge_reasoning": judge_raw,
    }
    return verdict, traj_data


def main():
    parser = argparse.ArgumentParser(description="Fast prompt tuning for Gemini teacher models")
    parser.add_argument("--model", default="gemini-2.0-flash-001", help="Model to test (default: gemini-2.0-flash-001)")
    parser.add_argument("--categories", nargs="+", default=None, help="Filter to categories (substring match)")
    parser.add_argument("--calendars", nargs="+", type=int, default=None, help="Calendar indices (default: 0 2 3 4 5)")
    parser.add_argument("--prompt-file", default=None, help="Path to custom system prompt .txt file")
    parser.add_argument("--prompt-id", default=None, help="Label for this prompt variant")
    args = parser.parse_args()

    calendars = args.calendars or DEFAULT_CALENDARS
    prompt_id = args.prompt_id or (os.path.basename(args.prompt_file).replace(".txt", "") if args.prompt_file else "default")

    # Load custom prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            system_prompt = f.read().strip()
        print(f"Prompt: {args.prompt_file} ({len(system_prompt)} chars)")
    else:
        system_prompt = SYSTEM_PROMPT
        print(f"Prompt: default SYSTEM_PROMPT ({len(system_prompt)} chars)")

    # Init credentials
    with open(str(CREDENTIALS_PATH)) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"],
        client_id=cd["client_id"], client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
    refresh_req = google.auth.transport.requests.Request()

    eval_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])
    gen_model = GenerativeModel(args.model, tools=[CALENDAR_TOOL], system_instruction=[system_prompt])

    # Collect tasks
    tasks = []
    for cal_idx in calendars:
        query_path = os.path.join(QUERY_DIR, f"{cal_idx}.txt")
        cal_path = os.path.join(JSON_CALENDAR_DIR, f"{cal_idx}.txt")
        if not os.path.exists(query_path) or not os.path.exists(cal_path):
            print(f"Skipping cal {cal_idx}: missing files")
            continue
        queries = json.load(open(query_path))
        for qi, q in enumerate(queries):
            tasks.append({"cal_idx": cal_idx, "qi": qi, "query": q, "cal_path": cal_path})

    # Filter by category
    if args.categories:
        tasks = [t for t in tasks
                 if any(cf.lower() in t["query"].get("category", "").lower() for cf in args.categories)]

    print(f"\nModel: {args.model}")
    print(f"Prompt ID: {prompt_id}")
    print(f"Calendars: {calendars}")
    print(f"Categories: {args.categories or 'all'}")
    print(f"Queries: {len(tasks)}")
    print()

    if not tasks:
        print("No matching queries found!")
        return

    # Run
    cat_results = defaultdict(lambda: {"correct": 0, "total": 0, "queries": []})
    total_correct = 0

    # Save trajectories to per-run directory
    run_dir = os.path.join(SFT_DATA_DIR, "tuning_runs", prompt_id)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    log_file = open(log_path, "w")
    log_file.write(f"Model: {args.model}\nPrompt: {args.prompt_file}\nPrompt ID: {prompt_id}\n")
    log_file.write(f"Calendars: {calendars}\nCategories: {args.categories or 'all'}\nQueries: {len(tasks)}\n")
    log_file.write("=" * 80 + "\n\n")
    log_file.flush()

    for i, task in enumerate(tasks):
        # Refresh token if needed
        if not creds.valid or creds.expired:
            creds.refresh(refresh_req)
            vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
            gen_model = GenerativeModel(args.model, tools=[CALENDAR_TOOL], system_instruction=[system_prompt])
            eval_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])
            print("  [TOKEN REFRESHED]")

        q = task["query"]
        cat = q.get("category", "unknown")
        cat_short = cat.split("(")[0].strip()

        print(f"  [{i+1}/{len(tasks)}] cal={task['cal_idx']} [{cat_short[:20]}] {q['query'][:50]}...", end=" ", flush=True)

        max_attempts = 3
        verdict = "Incorrect"
        traj_data = None
        for attempt in range(1, max_attempts + 1):
            attempt_traj = None
            try:
                attempt_verdict, attempt_traj = run_single(gen_model, eval_model, task["cal_path"], q)
            except Exception as e:
                attempt_verdict = "Error"
                print(f"ERROR: {type(e).__name__}: {e}", flush=True)
                import traceback
                traceback.print_exc()

            # Log every attempt
            if attempt_traj:
                log_file.write(f"[{i+1}/{len(tasks)}] cal={task['cal_idx']} q={task['qi']} | {cat_short} | attempt {attempt}/{max_attempts} | {attempt_verdict}\n")
                log_file.write(f"  Query: {q['query']}\n")
                log_file.write(f"  Expected: {q.get('expected_behavior', '')}\n")
                for step in attempt_traj["trajectory"]:
                    if step["role"] == "user":
                        log_file.write(f"  USER: {step['content']}\n")
                    elif step["role"] == "assistant":
                        log_file.write(f"  ASSISTANT: {step['content']}\n")
                    elif step["role"] == "tool_call":
                        args_str = json.dumps(step["args"], default=str, indent=2)
                        result_str = json.dumps(step["result"], default=str, indent=2)
                        log_file.write(f"  TOOL: {step['name']}\n")
                        log_file.write(f"    ARGS: {args_str}\n")
                        log_file.write(f"    RESULT: {result_str}\n")
                log_file.write(f"  JUDGE: {attempt_traj.get('judge_reasoning', '')}\n")
                log_file.write(f"  VERDICT: {attempt_verdict}\n")
                log_file.write(f"{'─'*80}\n\n")
                log_file.flush()

            if attempt_verdict == "Correct":
                verdict = "Correct"
                traj_data = attempt_traj
                if attempt > 1:
                    print(f"Correct (attempt {attempt})", flush=True)
                else:
                    print("Correct", flush=True)
                break
            else:
                traj_data = attempt_traj  # keep last attempt for reference
                if attempt < max_attempts:
                    print(f"retry {attempt}...", end=" ", flush=True)
                    time.sleep(0.3)
                else:
                    print(f"Incorrect ({max_attempts}/{max_attempts})", flush=True)

        cat_results[cat]["total"] += 1
        cat_results[cat]["queries"].append({
            "cal": task["cal_idx"], "qi": task["qi"],
            "query": q["query"][:80], "verdict": verdict,
        })
        if verdict == "Correct":
            cat_results[cat]["correct"] += 1
            total_correct += 1

        time.sleep(0.3)

    log_file.close()
    print(f"\nRun log saved to {log_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"RESULTS: {prompt_id} | {args.model}")
    print(f"{'='*60}")
    print(f"Overall: {total_correct}/{len(tasks)} ({total_correct/len(tasks)*100:.1f}%)")
    print()

    for cat in sorted(cat_results):
        r = cat_results[cat]
        pct = r["correct"] / r["total"] * 100 if r["total"] > 0 else 0
        short = cat.split("(")[0].strip()
        print(f"  {short:<40} {r['correct']}/{r['total']} ({pct:.0f}%)")
        # Show individual query verdicts
        for qr in r["queries"]:
            mark = "+" if qr["verdict"] == "Correct" else "-"
            print(f"    {mark} cal={qr['cal']} {qr['query'][:60]}")

    # Log results
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt_id": prompt_id,
        "model": args.model,
        "categories_filter": args.categories,
        "calendars": calendars,
        "total_correct": total_correct,
        "total": len(tasks),
        "accuracy": round(total_correct / len(tasks) * 100, 1),
        "per_category": {
            cat: {"correct": r["correct"], "total": r["total"]}
            for cat, r in cat_results.items()
        },
    }

    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    print(f"\nLogged to {LOG_PATH}")


if __name__ == "__main__":
    main()
