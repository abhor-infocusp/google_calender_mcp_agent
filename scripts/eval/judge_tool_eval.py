#!/usr/bin/env python3
"""Run the tool-using Gemini judge on the full 285-trajectory manual oracle.

Reuses the tools defined in judge_tool_sim.py. Each case is a fresh chat
session; the judge calls tools until it commits to a verdict. Records:

  - verdict (Correct | Incorrect | Unknown)
  - turn count (number of tool-calling rounds)
  - tool calls made (names, args)
  - total input + output tokens (summed across turns)
  - wall time

Then reports overall accuracy, per-cat accuracy, and cost/latency overhead
vs the context-fed Gemini structured baseline (single-call).

Usage:
    PYTHONPATH=src python scripts/eval/judge_tool_eval.py
"""
from __future__ import annotations
import json
import sys
import time
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import google.auth.transport.requests
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import (
    Part, GenerationConfig, GenerativeModel,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
from calendar_agent.paths import CREDENTIALS_PATH
from calendar_agent.judge.features import parse_day_state
from judge_tool_sim import (
    TOOLS, SYSTEM_PROMPT,
    search_events, list_day, get_event_attendees, get_event_time, get_calendar_diff,
)

PROJECT = "internal-ml-exp"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-001"
GEN_CFG = GenerationConfig(temperature=0.0, top_p=1.0, max_output_tokens=2048)
MAX_TURNS = 10
CONCURRENCY = 8

INPUT_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
TRUTH_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts_relabeled.jsonl"

# Gemini-2.0-flash pricing (USD per million tokens, approx as of 2026-05).
PRICE_IN_PER_M  = 0.075   # input tokens
PRICE_OUT_PER_M = 0.30    # output tokens


def init_vertex():
    cd = json.load(open(CREDENTIALS_PATH))
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"],
        client_id=cd["client_id"], client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project=PROJECT, location=LOCATION, credentials=creds)


def extract_verdict(text: str) -> str:
    last_lines = [l.strip().rstrip(".!?,;:") for l in text.splitlines() if l.strip()]
    for l in reversed(last_lines):
        ll = l.lower()
        if ll == "incorrect" or ll.endswith(" incorrect"):
            return "Incorrect"
        if ll == "correct" or (ll.endswith(" correct") and "incorrect" not in ll):
            return "Correct"
    return "Unknown"


def run_one(rec: dict) -> dict:
    """One judging session against one record. Thread-safe (own chat object)."""
    rec_state = {
        "before": parse_day_state(rec.get("before") or ""),
        "after":  parse_day_state(rec.get("after") or ""),
    }
    user_msg = (
        f"USER QUERY: {rec['query']}\n\n"
        f"EXPECTED (reference interpretation, may be over-specific): "
        f"{rec.get('expected') or '(not specified)'}\n\n"
        f"AGENT RESPONSE: {rec.get('final') or '(no response)'}\n\n"
        f"Use tools to verify the response against the calendar, then judge."
    )
    model = GenerativeModel(MODEL, system_instruction=[SYSTEM_PROMPT], tools=[TOOLS])
    chat = model.start_chat()

    t0 = time.time()
    tool_calls: list[dict] = []
    in_tokens_total = 0
    out_tokens_total = 0

    try:
        response = chat.send_message(user_msg, generation_config=GEN_CFG)
    except Exception as e:
        return {**_base(rec), "error": str(e)[:200], "verdict": "error",
                "turns": 0, "tool_calls": 0, "in_tokens": 0, "out_tokens": 0,
                "latency_s": time.time() - t0, "raw": ""}

    final_text = ""
    for turn in range(MAX_TURNS):
        # Sum tokens from this response
        meta = getattr(response, "usage_metadata", None)
        if meta:
            in_tokens_total += int(getattr(meta, "prompt_token_count", 0) or 0)
            out_tokens_total += int(getattr(meta, "candidates_token_count", 0) or 0)

        cand = response.candidates[0]
        function_calls = []
        text_parts = []
        for part in cand.content.parts:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                function_calls.append(fc)
                continue
            try:
                if part.text:
                    text_parts.append(part.text)
            except Exception:
                pass
        text = "".join(text_parts).strip()

        if function_calls:
            tool_responses = []
            for fc in function_calls:
                args = dict(fc.args)
                name = fc.name
                when = args.get("when", "before")
                try:
                    if name == "search_events":
                        result = search_events(rec_state, args.get("keywords", ""), args.get("day"), when)
                    elif name == "list_day":
                        result = list_day(rec_state, args.get("day", ""), when)
                    elif name == "get_event_attendees":
                        result = get_event_attendees(rec_state, args.get("title", ""), args.get("day", ""), when)
                    elif name == "get_event_time":
                        result = get_event_time(rec_state, args.get("title", ""), args.get("day", ""), when)
                    elif name == "get_calendar_diff":
                        result = get_calendar_diff(rec_state)
                    else:
                        result = {"error": f"unknown tool {name}"}
                except Exception as e:
                    result = {"error": str(e)[:200]}
                tool_calls.append({"turn": turn, "name": name, "args": args, "result_keys": list(result.keys())})
                tool_responses.append(Part.from_function_response(
                    name=name, response={"content": result},
                ))
            try:
                response = chat.send_message(tool_responses, generation_config=GEN_CFG)
            except Exception as e:
                final_text = f"[chat send error] {e}"
                break
            continue

        # No function calls — final answer
        final_text = text
        break
    else:
        final_text = "[max turns reached]"

    verdict = extract_verdict(final_text)
    return {
        **_base(rec),
        "verdict": verdict,
        "turns": len(tool_calls),
        "tool_calls": [tc["name"] for tc in tool_calls],
        "tool_call_log": tool_calls,
        "in_tokens": in_tokens_total,
        "out_tokens": out_tokens_total,
        "latency_s": time.time() - t0,
        "raw": final_text[:1500],
        "error": "",
    }


def _base(rec):
    return {"sid": rec["sid"], "cat": rec["cat"], "gt": rec["gt"]}


def main():
    init_vertex()
    inputs = [json.loads(l) for l in INPUT_JSONL.open()]
    truth = [json.loads(l) for l in TRUTH_JSONL.open()]
    cases = []
    for i, r in enumerate(inputs):
        if i >= len(truth): continue
        rec = dict(r)
        rec["gt"] = truth[i]["verdict"]
        cases.append(rec)
    print(f"loaded {len(cases)} cases")

    out_dir = REPO / f"runs/judge_tool_eval_{datetime.now():%Y%m%d_%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(run_one, rec): rec for rec in cases}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 25 == 0 or i == len(cases):
                print(f"  {i}/{len(cases)} done in {time.time()-t0:.1f}s")

    # Aggregate
    n = len(results)
    n_err = sum(1 for r in results if r.get("error") or r["verdict"] == "error")
    n_unknown = sum(1 for r in results if r["verdict"] == "Unknown")
    n_correct = sum(1 for r in results if r["verdict"] == r["gt"])

    by_cat = defaultdict(lambda: {"n": 0, "right": 0})
    by_turns = Counter()
    tool_use = Counter()
    in_tok_total = out_tok_total = 0
    latencies = []

    for r in results:
        c = r["cat"]
        by_cat[c]["n"] += 1
        if r["verdict"] == r["gt"]:
            by_cat[c]["right"] += 1
        by_turns[r["turns"]] += 1
        for tc in r["tool_calls"]:
            tool_use[tc] += 1
        in_tok_total += r["in_tokens"]
        out_tok_total += r["out_tokens"]
        latencies.append(r["latency_s"])

    cost_in = in_tok_total / 1e6 * PRICE_IN_PER_M
    cost_out = out_tok_total / 1e6 * PRICE_OUT_PER_M
    cost_total = cost_in + cost_out
    latencies.sort()
    p50 = latencies[len(latencies)//2] if latencies else 0
    p90 = latencies[int(len(latencies)*0.9)] if latencies else 0

    print()
    print("=" * 70)
    print(f"=== TOOL-USING JUDGE — {MODEL} on relabeled gt ===")
    print("=" * 70)
    print(f"overall: {n_correct}/{n} = {100*n_correct/n:.2f}%   errors={n_err}  unknown={n_unknown}")
    print(f"latency p50={p50:.2f}s  p90={p90:.2f}s   wall {time.time()-t0:.1f}s @ concurrency={CONCURRENCY}")
    print()
    print("Per-category:")
    for c, d in sorted(by_cat.items(), key=lambda kv: -kv[1]["right"]/max(kv[1]["n"],1)):
        print(f"  {c[:55]:55s} {d['right']:>3d}/{d['n']:<3d} = {100*d['right']/max(d['n'],1):.2f}%")
    print()
    print(f"Tool-call distribution: turns/case histogram = {dict(by_turns)}")
    print(f"Tool usage frequency:")
    for tn, cnt in tool_use.most_common():
        print(f"  {tn}: {cnt}")
    print()
    print(f"Tokens: in={in_tok_total:,}  out={out_tok_total:,}  total={in_tok_total+out_tok_total:,}")
    print(f"Cost (gemini-2.0-flash): in=${cost_in:.4f}  out=${cost_out:.4f}  total=${cost_total:.4f}")
    print(f"Per-case cost: ${cost_total/n*1000:.4f}m  (×1000 = milli-dollars)")

    # Save
    (out_dir / "results.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in results)
    )
    (out_dir / "summary.json").write_text(json.dumps({
        "model": MODEL,
        "n": n, "n_correct": n_correct,
        "accuracy": n_correct / n,
        "errors": n_err, "unknown": n_unknown,
        "latency_p50": p50, "latency_p90": p90,
        "wall_time_s": time.time()-t0,
        "tokens": {"in": in_tok_total, "out": out_tok_total},
        "cost_usd": {"in": cost_in, "out": cost_out, "total": cost_total},
        "turns_histogram": dict(by_turns),
        "tool_use_counts": dict(tool_use),
        "per_category": {c: dict(d) for c, d in by_cat.items()},
    }, indent=2))
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
