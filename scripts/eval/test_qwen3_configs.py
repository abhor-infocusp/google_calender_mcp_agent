#!/usr/bin/env python3
"""Test Qwen3-14B with different configurations to find optimal setup.

Tests: thinking on/off, various system prompts, across diverse query types.
"""

import json
import time
import sys
import os

from openai import OpenAI
from calendar_agent.environment import CalendarEnvironment
from calendar_agent.core import (
    SYSTEM_PROMPT, dispatch_tool_call, format_tool_result,
    compute_fallback_now, snapshot_events, filter_by_days,
)
from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, format_day_state_text
from calendar_agent.tools import get_openai_tools, get_openai_tools_minimal
from calendar_agent.paths import RL_DATA_DIR

import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerativeModel
from calendar_agent.paths import CREDENTIALS_PATH

# ── Gemini judge setup ──
cred_path = str(CREDENTIALS_PATH)
if os.path.exists(cred_path):
    with open(cred_path) as f:
        cred_data = json.load(f)
    gcp_creds = OAuth2Credentials(
        token=None,
        refresh_token=cred_data["refresh_token"],
        client_id=cred_data["client_id"],
        client_secret=cred_data["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=gcp_creds)
    eval_model = GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])
else:
    eval_model = None
    print("WARNING: No credentials, skipping judge evaluation")


def judge_query(query, final_output, expected, before_days, after_days):
    """Ask Gemini judge for verdict."""
    if eval_model is None:
        return "NoJudge"
    before_text = format_day_state_text(before_days)
    after_text = format_day_state_text(after_days)
    prompt = f"""Query: {query}

Response: {final_output if final_output else '(no response)'}

Expected: {expected if expected else '(not specified)'}

Before:
{before_text}

After:
{after_text}

Was the task completed correctly? End with one word: Correct or Incorrect."""
    try:
        response = eval_model.generate_content(prompt)
        text = response.text.strip()
        for line in reversed(text.splitlines()):
            line = line.strip().lower()
            if line == "incorrect":
                return "Incorrect"
            if line == "correct":
                return "Correct"
        for line in reversed(text.splitlines()):
            if "incorrect" in line.lower():
                return "Incorrect"
            if "correct" in line.lower():
                return "Correct"
        return "Incorrect"
    except Exception as e:
        return f"Error({e})"


# ── Configurations to test ──

CONFIGS = {
    "think_no_sys": {
        "system_prompt": None,
        "tools": "full",
        "extra_body": None,
        "desc": "Thinking ON, no system prompt",
    },
    "think_full_sys": {
        "system_prompt": SYSTEM_PROMPT,
        "tools": "full",
        "extra_body": None,
        "desc": "Thinking ON, full system prompt",
    },
    "think_minimal_sys": {
        "system_prompt": "You are a calendar assistant. Use the provided tools to manage events. Call get_current_time first to know the current date.",
        "tools": "full",
        "extra_body": None,
        "desc": "Thinking ON, minimal system prompt",
    },
    "nothink_no_sys": {
        "system_prompt": "/no_think",
        "tools": "full",
        "extra_body": None,
        "desc": "Thinking OFF (/no_think), no real system prompt",
    },
    "nothink_full_sys": {
        "system_prompt": "/no_think\n" + SYSTEM_PROMPT,
        "tools": "full",
        "extra_body": None,
        "desc": "Thinking OFF, full system prompt",
    },
    "nothink_minimal_sys": {
        "system_prompt": "/no_think\nYou are a calendar assistant. Use the provided tools to manage events. Call get_current_time first to know the current date.",
        "tools": "full",
        "extra_body": None,
        "desc": "Thinking OFF, minimal system prompt",
    },
    "nothink_minimal_tools": {
        "system_prompt": "/no_think\nYou are a calendar assistant. Use the provided tools to manage events. Call get_current_time first to know the current date.",
        "tools": "minimal",
        "extra_body": None,
        "desc": "Thinking OFF, minimal system prompt, minimal tools",
    },
}

# ── Test queries (diverse categories, from RL data) ──

def load_test_queries():
    """Load a diverse set of test queries from RL data."""
    rl_dir = str(RL_DATA_DIR)
    query_dir = os.path.join(rl_dir, "queries")
    cal_dir = os.path.join(rl_dir, "json_calender")

    # Pick queries from different categories across a few calendars
    test_cases = []
    seen_categories = {}

    for cal_idx in range(10):  # First 10 calendars
        query_path = os.path.join(query_dir, f"{cal_idx}.txt")
        cal_path = os.path.join(cal_dir, f"{cal_idx}.txt")
        if not os.path.exists(query_path) or not os.path.exists(cal_path):
            continue
        queries = json.load(open(query_path))
        for qi, q in enumerate(queries):
            cat = q.get("category", "Unknown")
            # Take up to 2 per category
            if seen_categories.get(cat, 0) >= 2:
                continue
            seen_categories[cat] = seen_categories.get(cat, 0) + 1
            test_cases.append({
                "cal_idx": cal_idx,
                "query_idx": qi,
                "query": q["query"],
                "expected": q.get("expected_behavior", ""),
                "category": cat,
                "addressed_days": q.get("addressed_days", []),
                "current_time": q.get("current_time", ""),
                "cal_path": cal_path,
            })
        if len(test_cases) >= 14:
            break

    return test_cases


def run_agent(client, model_name, tools_list, system_prompt, env, query, max_turns=6):
    """Run agent loop, return (final_answer, num_turns, total_tokens, elapsed_s)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})

    start = time.monotonic()
    total_tokens = 0
    final_answer = None

    for turn in range(1, max_turns + 1):
        try:
            resp = client.chat.completions.create(
                model=model_name, messages=messages, tools=tools_list, temperature=0.7,
            )
        except Exception as e:
            final_answer = f"(error: {e})"
            break

        if resp.usage:
            total_tokens += (resp.usage.prompt_tokens or 0) + (resp.usage.completion_tokens or 0)

        msg = resp.choices[0].message
        asst = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            asst["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(asst)

        if not msg.tool_calls:
            content = msg.content or ""
            # Strip think blocks from final answer
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            final_answer = content
            break

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = dispatch_tool_call(env, tc.function.name, args)
            result_str = format_tool_result(result)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str})

    elapsed = time.monotonic() - start
    return final_answer, turn, total_tokens, elapsed


def main():
    client = OpenAI(base_url="http://localhost:8005/v1", api_key="x")
    full_tools = get_openai_tools()
    minimal_tools = get_openai_tools_minimal()

    test_cases = load_test_queries()
    print(f"Loaded {len(test_cases)} test queries across categories:")
    cats = {}
    for tc in test_cases:
        cats[tc["category"]] = cats.get(tc["category"], 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print()

    # ── Run all configs ──
    results = {}
    for config_name, config in CONFIGS.items():
        print(f"\n{'='*70}")
        print(f"CONFIG: {config_name} — {config['desc']}")
        print(f"{'='*70}")

        tools_list = full_tools if config["tools"] == "full" else minimal_tools
        config_results = []

        for i, tc in enumerate(test_cases):
            # Fresh environment per query
            env = CalendarEnvironment()
            events = CalendarEnvironment.load_json_calendar(tc["cal_path"])
            fallback_now = compute_fallback_now(tc["cal_path"])
            ct = tc["current_time"].replace("T", " ") if tc["current_time"] else fallback_now
            env.initialize(events=events, now=ct)

            before_snap = snapshot_events(env)

            answer, turns, tokens, elapsed = run_agent(
                client, "qwen3-14b", tools_list,
                config["system_prompt"], env, tc["query"],
            )

            after_snap = snapshot_events(env)
            days = tc["addressed_days"] if tc["addressed_days"] else list(before_snap.keys())
            before_days = filter_by_days(before_snap, days)
            after_days = filter_by_days(after_snap, days)

            verdict = judge_query(tc["query"], answer, tc["expected"], before_days, after_days)

            cat_short = tc["category"][:30]
            answer_short = (answer or "(none)")[:80].replace("\n", " ")
            print(f"  [{i+1}/{len(test_cases)}] [{cat_short}] {verdict:10s} "
                  f"turns={turns} tok={tokens:4d} {elapsed:5.1f}s | {tc['query'][:50]}")

            config_results.append({
                "query": tc["query"],
                "category": tc["category"],
                "verdict": verdict,
                "turns": turns,
                "tokens": tokens,
                "elapsed_s": round(elapsed, 1),
                "answer_preview": answer_short,
            })

        # ── Summary for this config ──
        correct = sum(1 for r in config_results if r["verdict"] == "Correct")
        total = len(config_results)
        avg_turns = sum(r["turns"] for r in config_results) / total
        avg_tokens = sum(r["tokens"] for r in config_results) / total
        avg_time = sum(r["elapsed_s"] for r in config_results) / total

        print(f"\n  SUMMARY: {correct}/{total} ({correct/total*100:.0f}%) correct | "
              f"avg {avg_turns:.1f} turns, {avg_tokens:.0f} tokens, {avg_time:.1f}s/query")

        # Per-category breakdown
        cat_stats = {}
        for r in config_results:
            cat = r["category"]
            if cat not in cat_stats:
                cat_stats[cat] = {"correct": 0, "total": 0}
            cat_stats[cat]["total"] += 1
            if r["verdict"] == "Correct":
                cat_stats[cat]["correct"] += 1
        for cat, s in sorted(cat_stats.items()):
            print(f"    {cat[:40]:40s} {s['correct']}/{s['total']}")

        results[config_name] = {
            "desc": config["desc"],
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 3),
            "avg_turns": round(avg_turns, 1),
            "avg_tokens": round(avg_tokens, 0),
            "avg_time_s": round(avg_time, 1),
            "per_query": config_results,
        }

    # ── Final comparison ──
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<30s} {'Acc':>6s} {'Turns':>6s} {'Tokens':>7s} {'Time':>6s}")
    print("-" * 60)
    for config_name, r in results.items():
        print(f"{config_name:<30s} {r['accuracy']*100:5.0f}% {r['avg_turns']:5.1f} "
              f"{r['avg_tokens']:6.0f} {r['avg_time_s']:5.1f}s")

    # Save full results
    with open("eval_qwen3_config_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to eval_qwen3_config_comparison.json")


if __name__ == "__main__":
    main()
