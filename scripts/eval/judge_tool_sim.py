#!/usr/bin/env python3
"""Tool-calling simulation: let Gemini-as-judge query for the info it wants.

Picks two IR cases (one Gemini currently fails, one it currently passes),
gives Gemini a small toolbox it can call to inspect the calendar, and runs
a multi-turn conversation. Captures which tools it chose, what arguments,
and what verdict it landed on. Useful as a Tier-2 prototype before deciding
whether to build a full tool-using judge.

Tools exposed to Gemini:
    search_events(keywords, day=None)   → events matching keywords
    list_day(day)                       → all events on that day
    get_event_attendees(title, day)     → attendee emails of one event
    get_event_time(title, day)          → start-end time of one event

Each tool reads from the BEFORE-state events (parsed from the formatted
day-state text via features.parse_day_state). We never feed BEFORE/AFTER
text dumps — the judge has to ask.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import google.auth.transport.requests
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import (
    Content, Part, FunctionDeclaration, Tool,
    GenerationConfig, GenerativeModel,
)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from calendar_agent.paths import CREDENTIALS_PATH
from calendar_agent.judge.features import parse_day_state, compute_diff, fmt_diff, Event

PROJECT = "internal-ml-exp"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-001"
GEN_CFG = GenerationConfig(temperature=0.0, top_p=1.0, max_output_tokens=2048)


def init_vertex():
    cd = json.load(open(CREDENTIALS_PATH))
    creds = OAuth2Credentials(
        token=None, refresh_token=cd["refresh_token"],
        client_id=cd["client_id"], client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project=PROJECT, location=LOCATION, credentials=creds)


# ── Tool implementations ──
# Tools accept a `when` parameter selecting before|after state.
def _events_for(rec_state: dict, when: str) -> list[Event]:
    return rec_state["after" if when == "after" else "before"]


def search_events(rec_state: dict, keywords: str, day: str | None = None,
                  when: str = "before") -> dict:
    events = _events_for(rec_state, when)
    kws = [k.lower() for k in keywords.split() if k]
    matches = []
    for e in events:
        if day and e.day.lower() != day.lower():
            continue
        haystack = (e.title + " " + " ".join(e.attendees)).lower()
        score = sum(1 for k in kws if k in haystack)
        if score:
            matches.append({
                "title": e.title, "day": e.day,
                "start": e.start, "end": e.end,
                "attendees": list(e.attendees),
                "match_score": score,
            })
    matches.sort(key=lambda m: -m["match_score"])
    return {"matches": matches[:5], "state": when}


def list_day(rec_state: dict, day: str, when: str = "before") -> dict:
    events = _events_for(rec_state, when)
    out = [
        {"title": e.title, "start": e.start, "end": e.end,
         "attendees": list(e.attendees)}
        for e in events if e.day.lower() == day.lower()
    ]
    return {"day": day, "state": when, "events": out}


def get_calendar_diff(rec_state: dict) -> dict:
    """Return a structured diff between BEFORE and AFTER states."""
    diff = compute_diff(rec_state["before"], rec_state["after"])
    return {
        "added":    [{"title": e.title, "day": e.day, "start": e.start, "end": e.end,
                      "attendees": list(e.attendees)} for e in diff.added],
        "removed":  [{"title": e.title, "day": e.day, "start": e.start, "end": e.end,
                      "attendees": list(e.attendees)} for e in diff.removed],
        "modified": [{"title": old.title, "from_day": old.day,
                      "from": f"{old.start}-{old.end}", "to": f"{new.start}-{new.end}",
                      "fields_changed": fields}
                     for old, new, fields in diff.modified],
        "moved":    [{"title": old.title, "from_day": old.day,
                      "from_time": f"{old.start}-{old.end}",
                      "to_day": new.day, "to_time": f"{new.start}-{new.end}"}
                     for old, new in diff.moved],
        "is_empty": diff.is_empty(),
    }


def _find_one(events, title, day):
    title_l = title.lower()
    cands = [e for e in events
             if e.day.lower() == day.lower() and title_l in e.title.lower()]
    if not cands:
        # try fuzzier — any keyword in title
        kws = [k for k in title_l.split() if len(k) > 2]
        cands = [e for e in events
                 if e.day.lower() == day.lower()
                 and any(k in e.title.lower() for k in kws)]
    return cands[0] if cands else None


def get_event_attendees(rec_state: dict, title: str, day: str, when: str = "before") -> dict:
    events = _events_for(rec_state, when)
    e = _find_one(events, title, day)
    if not e:
        return {"error": f"no event matching title '{title}' on {day} ({when})", "state": when}
    return {"title": e.title, "day": e.day, "attendees": list(e.attendees) or "NONE", "state": when}


def get_event_time(rec_state: dict, title: str, day: str, when: str = "before") -> dict:
    events = _events_for(rec_state, when)
    e = _find_one(events, title, day)
    if not e:
        return {"error": f"no event matching title '{title}' on {day} ({when})", "state": when}
    return {"title": e.title, "day": e.day, "start": e.start, "end": e.end, "state": when}


# ── Tool declarations for Gemini ──
_WHEN_DESC = ("Which calendar state to query: 'before' (state before the agent acted) "
              "or 'after' (state after the agent acted). Defaults to 'before'. "
              "Use 'after' to verify what the agent actually changed.")

TOOLS = Tool(function_declarations=[
    FunctionDeclaration(
        name="search_events",
        description="Search events by keywords. Optionally filter by day. Searches the BEFORE or AFTER state.",
        parameters={
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "Words to look for in event titles/attendees."},
                "day": {"type": "string", "description": "Optional day-of-week filter."},
                "when": {"type": "string", "description": _WHEN_DESC},
            },
            "required": ["keywords"],
        },
    ),
    FunctionDeclaration(
        name="list_day",
        description="List every event scheduled on a given day-of-week, in BEFORE or AFTER state.",
        parameters={
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "Day of week."},
                "when": {"type": "string", "description": _WHEN_DESC},
            },
            "required": ["day"],
        },
    ),
    FunctionDeclaration(
        name="get_event_attendees",
        description="Return the attendee email list of one specific event in BEFORE or AFTER state.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title or substring."},
                "day": {"type": "string", "description": "Day of week."},
                "when": {"type": "string", "description": _WHEN_DESC},
            },
            "required": ["title", "day"],
        },
    ),
    FunctionDeclaration(
        name="get_event_time",
        description="Return the start and end time of one specific event in BEFORE or AFTER state.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Event title or substring."},
                "day": {"type": "string", "description": "Day of week."},
                "when": {"type": "string", "description": _WHEN_DESC},
            },
            "required": ["title", "day"],
        },
    ),
    FunctionDeclaration(
        name="get_calendar_diff",
        description=("Return a structured diff between the BEFORE and AFTER calendar states: "
                     "events added, removed, modified (in-place edit), and moved (cross-day relocation). "
                     "Use this when you need to verify what the agent changed."),
        parameters={"type": "object", "properties": {}},
    ),
])


SYSTEM_PROMPT = """\
You are a judge evaluating a calendar assistant. The user gave a query;
the agent responded (and may have modified the calendar). Decide whether
the agent completed the task correctly.

You CANNOT see the calendar directly. You have these tools:
  - search_events(keywords, day?, when?)
  - list_day(day, when?)
  - get_event_attendees(title, day, when?)
  - get_event_time(title, day, when?)
  - get_calendar_diff()    — structured BEFORE→AFTER diff

The `when` argument selects 'before' (pre-action state) or 'after'
(post-action state). For information-retrieval queries the calendar
should not change, so 'before' is enough. For action queries
(schedule/modify/delete), use get_calendar_diff() to see what changed
and verify it matches the user's intent.

Trust the QUERY over the EXPECTED — EXPECTED is one approximate
interpretation, not gospel. If the agent did what the user asked but
not what EXPECTED specifies, lean Correct.

Use tools to gather whatever facts you need. Be thorough but efficient.
Once confident, output your reasoning followed by one final word on the
last line: Correct or Incorrect.
"""


def run_case(model, rec, rec_state):
    """Multi-turn conversation: judge calls tools, harness responds."""
    user_msg = (
        f"USER QUERY: {rec['query']}\n\n"
        f"EXPECTED (reference interpretation, may be over-specific): {rec.get('expected') or '(not specified)'}\n\n"
        f"AGENT RESPONSE: {rec['final']}\n\n"
        f"Use tools to verify the response against the calendar, then judge."
    )

    chat = model.start_chat()
    print(f"\n{'='*70}\nCASE: {rec['sid']}  (gt={rec['gt']})")
    print(f"QUERY:    {rec['query']}")
    print(f"RESPONSE: {rec['final'][:200]}")
    print('='*70)

    response = chat.send_message(user_msg)

    transcript = []
    max_turns = 8
    for turn in range(max_turns):
        # Look for function calls in response
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
            # Execute each function call, collect responses
            tool_responses = []
            for fc in function_calls:
                args = dict(fc.args)
                name = fc.name
                when = args.get("when", "before")
                if name == "search_events":
                    result = search_events(rec_state, args.get("keywords",""), args.get("day"), when)
                elif name == "list_day":
                    result = list_day(rec_state, args.get("day",""), when)
                elif name == "get_event_attendees":
                    result = get_event_attendees(rec_state, args.get("title",""), args.get("day",""), when)
                elif name == "get_event_time":
                    result = get_event_time(rec_state, args.get("title",""), args.get("day",""), when)
                elif name == "get_calendar_diff":
                    result = get_calendar_diff(rec_state)
                else:
                    result = {"error": f"unknown tool {name}"}
                print(f"\n[turn {turn}] TOOL CALL: {name}({json.dumps(args)})")
                print(f"           RESULT: {json.dumps(result)[:300]}")
                transcript.append({"turn": turn, "type": "tool_call",
                                   "name": name, "args": args, "result": result})
                tool_responses.append(Part.from_function_response(
                    name=name, response={"content": result},
                ))
            response = chat.send_message(tool_responses)
            continue

        # No function calls — final answer
        print(f"\n[turn {turn}] FINAL TEXT:")
        print(text)
        transcript.append({"turn": turn, "type": "final", "text": text})
        # Extract verdict
        # Strip trailing punctuation when matching verdict
        last_lines = [l.strip().rstrip(".!?,;:") for l in text.splitlines() if l.strip()]
        verdict = "Unknown"
        for l in reversed(last_lines):
            ll = l.lower()
            if ll == "incorrect" or ll.endswith(" incorrect"):
                verdict = "Incorrect"; break
            if ll == "correct" or (ll.endswith(" correct") and "incorrect" not in ll):
                verdict = "Correct"; break
        print(f"\nverdict: {verdict}   gt: {rec['gt']}   match: {verdict == rec['gt']}")
        return {"sid": rec["sid"], "gt": rec["gt"], "verdict": verdict,
                "transcript": transcript}

    print("\n[max turns hit without verdict]")
    return {"sid": rec["sid"], "gt": rec["gt"], "verdict": "max_turns",
            "transcript": transcript}


def main():
    init_vertex()
    inputs = [json.loads(l) for l in
              (REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl").open()]
    truth = [json.loads(l) for l in
             (REPO / "runs/judge_baseline_20260430/eval/manual_verdicts_relabeled.jsonl").open()]

    # All 5 sids that I flipped during the relabel pass — re-verify with tools.
    # If Gemini-with-tools agrees with my flipped verdict, the flip stands.
    # If it agrees with the ORIGINAL gt, the flip was wrong.
    targets = ["cal_32_q_7", "cal_8_q_8", "cal_19_q_8", "cal_19_q_1", "cal_22_q_2"]
    # Also load original gt for comparison
    truth_orig = [json.loads(l) for l in
                  (REPO / "runs/judge_baseline_20260430/eval/manual_verdicts.jsonl").open()]

    cases = []
    for sid in targets:
        for i, r in enumerate(inputs):
            if r["sid"] == sid:
                rec = dict(r)
                rec["gt"] = truth[i]["verdict"]                  # the (possibly flipped) relabeled gt
                rec["gt_original"] = truth_orig[i]["verdict"]    # the original manual gt
                cases.append(rec)
                break

    model = GenerativeModel(MODEL, system_instruction=[SYSTEM_PROMPT], tools=[TOOLS])

    out_dir = REPO / "runs/judge_tool_sim_relabel_audit_20260507"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for rec in cases:
        rec_state = {
            "before": parse_day_state(rec.get("before") or ""),
            "after":  parse_day_state(rec.get("after") or ""),
        }
        result = run_case(model, rec, rec_state)
        result["gt_original"] = rec["gt_original"]
        result["gt_relabeled"] = rec["gt"]
        result["query"] = rec["query"]
        result["final"] = rec["final"]
        results.append(result)

    print("\n" + "=" * 70)
    print("RELABEL AUDIT SUMMARY")
    print("=" * 70)
    print(f"{'sid':12s} {'orig gt':10s} {'my flip':10s} {'tool-Gem':10s} {'flip ok?'}")
    print("-" * 70)
    for r in results:
        flip_ok = "YES" if r["verdict"] == r["gt_relabeled"] else "NO (revert)" if r["verdict"] == r["gt_original"] else "?"
        print(f"{r['sid']:12s} {r['gt_original']:10s} {r['gt_relabeled']:10s} {r['verdict']:10s} {flip_ok}")

    (out_dir / "transcripts.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[saved transcripts → {out_dir / 'transcripts.json'}]")


if __name__ == "__main__":
    main()
