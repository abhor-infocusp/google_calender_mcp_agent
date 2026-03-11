#!/usr/bin/env python3
"""Benchmark Gemini models on high-complexity queries to find the cheapest one
that consistently solves them.

Tests models from cheapest to most expensive, running trajectories on all
high-complexity queries and evaluating correctness.
"""

import json
import glob
import os
import sys
import uuid
import warnings

warnings.filterwarnings("ignore")

import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool

from environment.environment import CalendarEnvironment
from run_trajectory import (
    CALENDAR_TOOL,
    SYSTEM_PROMPT,
    EVAL_SYSTEM_PROMPT,
    TOOL_DECLARATIONS,
    dispatch_tool_call,
    snapshot_events,
    filter_by_days,
    evaluate_trajectory,
    get_query_now,
    DAY_NAMES,
)

# Models to test, cheapest first
MODELS_TO_TEST = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

SFT_DATA_DIR = os.path.join(os.path.dirname(__file__), "sft_data")
JSON_CALENDAR_DIR = os.path.join(SFT_DATA_DIR, "json_calender")
QUERY_DIR = os.path.join(SFT_DATA_DIR, "queries")

CONSISTENCY_THRESHOLD = 0.8  # 80% correct to be "consistent"


def load_high_complexity_queries():
    """Load all high complexity queries with their calendar indices."""
    high_queries = []
    for f in sorted(glob.glob(os.path.join(QUERY_DIR, "*.txt"))):
        cal_idx = os.path.basename(f).replace(".txt", "")
        cal_path = os.path.join(JSON_CALENDAR_DIR, f"{cal_idx}.txt")
        if not os.path.exists(cal_path):
            continue
        data = json.load(open(f))
        for qi, q in enumerate(data):
            if q.get("complexity") == "High":
                high_queries.append({
                    "cal_idx": cal_idx,
                    "query_idx": qi,
                    "query": q,
                    "cal_path": cal_path,
                })
    return high_queries


def run_single_query(model, eval_model, cal_path, query_dict, max_turns=10):
    """Run a single query and return the verdict."""
    events = CalendarEnvironment.load_json_calendar(cal_path)

    # Derive fallback now
    from datetime import datetime
    earliest = None
    for evt in events:
        dt = datetime.fromisoformat(evt["start"])
        if earliest is None or dt < earliest:
            earliest = dt
    fallback_now = earliest.replace(hour=8, minute=0, second=0).strftime("%Y-%m-%d %H:%M:%S")

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
        return "Incorrect", str(e)

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
            if result is None:
                result = {"status": "ok"}
            result = json.loads(json.dumps(result, default=str))
            trajectory.append({
                "role": "tool_call",
                "name": fc.name,
                "args": args,
                "result": result,
            })
            response_parts.append(
                Part.from_function_response(name=fc.name, response=result)
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
    return verdict, final_output


def test_model(model_name, high_queries, credentials):
    """Test a model on all high complexity queries. Returns (correct, total, results)."""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")

    try:
        model = GenerativeModel(
            model_name,
            tools=[CALENDAR_TOOL],
            system_instruction=[SYSTEM_PROMPT],
        )
        eval_model = GenerativeModel(
            "gemini-2.0-flash-001",  # Always use flash for eval
            system_instruction=[EVAL_SYSTEM_PROMPT],
        )
    except Exception as e:
        print(f"  Failed to create model: {e}")
        return 0, len(high_queries), []

    results = []
    correct = 0
    total = len(high_queries)

    for i, hq in enumerate(high_queries):
        cal_idx = hq["cal_idx"]
        qi = hq["query_idx"]
        query_text = hq["query"]["query"][:60]
        print(f"  [{i+1}/{total}] cal={cal_idx} q={qi}: {query_text}...", end=" ", flush=True)

        try:
            verdict, _ = run_single_query(model, eval_model, hq["cal_path"], hq["query"])
            results.append({"cal_idx": cal_idx, "query_idx": qi, "verdict": verdict})
            if verdict == "Correct":
                correct += 1
            print(verdict)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"cal_idx": cal_idx, "query_idx": qi, "verdict": "Error"})

    rate = correct / total if total > 0 else 0
    print(f"\n  Result: {correct}/{total} correct ({rate:.1%})")
    return correct, total, results


def main():
    # Init credentials
    creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcloud_credentials.json")
    with open(creds_path) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None,
        refresh_token=cd["refresh_token"],
        client_id=cd["client_id"],
        client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)

    high_queries = load_high_complexity_queries()
    print(f"Loaded {len(high_queries)} high complexity queries")

    # Test models from cheapest to most expensive
    all_results = {}
    for model_name in MODELS_TO_TEST:
        correct, total, results = test_model(model_name, high_queries, creds)
        rate = correct / total if total > 0 else 0
        all_results[model_name] = {
            "correct": correct,
            "total": total,
            "rate": rate,
            "results": results,
        }

        if rate >= CONSISTENCY_THRESHOLD:
            print(f"\n*** {model_name} passes threshold ({rate:.1%} >= {CONSISTENCY_THRESHOLD:.0%}) ***")
            print(f"*** This is the cheapest consistent model. ***")
            break
        else:
            print(f"\n  {model_name} below threshold ({rate:.1%} < {CONSISTENCY_THRESHOLD:.0%}), trying next...")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for model_name, r in all_results.items():
        status = "PASS" if r["rate"] >= CONSISTENCY_THRESHOLD else "FAIL"
        print(f"  {model_name}: {r['correct']}/{r['total']} ({r['rate']:.1%}) [{status}]")

    # Save results
    results_path = os.path.join(SFT_DATA_DIR, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
