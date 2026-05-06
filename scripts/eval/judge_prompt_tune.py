#!/usr/bin/env python3
"""Iterate on judge prompts against the manual-labeled ART hold-out (285).

Server: vLLM serving Qwen3-14B base on http://azkaban:8011 (see
scripts/eval/judge_prompt_serve.sbatch).

Truth: runs/judge_baseline_20260430/eval/manual_verdicts.jsonl

Usage:
    PYTHONPATH=src python scripts/eval/judge_prompt_tune.py --variant baseline
    PYTHONPATH=src python scripts/eval/judge_prompt_tune.py --variant cot_checklist --concurrency 16
    PYTHONPATH=src python scripts/eval/judge_prompt_tune.py --list

Each run appends one row to runs/judge_prompt_tune_20260430/results/summary.csv
and saves per-trajectory predictions to results/<variant>.jsonl.
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

# ── Paths
REPO = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO / "runs/judge_baseline_20260430/eval"
INPUT_JSONL = EVAL_DIR / "manual_review_input.jsonl"
TRUTH_JSONL = EVAL_DIR / "manual_verdicts.jsonl"
OUT_DIR = REPO / "runs/judge_prompt_tune_20260430/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "summary.csv"

API_BASE = os.environ.get("JUDGE_API_BASE", "http://azkaban:8011/v1")
MODEL = os.environ.get("JUDGE_MODEL", "judge")

# Cache global recs / truth
RECS = [json.loads(l) for l in open(INPUT_JSONL)]
TRUTH = {int(json.loads(l)["idx"]): json.loads(l)["verdict"] for l in open(TRUTH_JSONL)}
assert len(RECS) == len(TRUTH) == 285


# ─────────────────────────────────────────────────────────────
# Verdict extraction (same logic as eval_judge_on_art.extract_verdict)
# ─────────────────────────────────────────────────────────────
def extract_verdict(text: str) -> str:
    if not text:
        return "Incorrect"
    # Strip <think>...</think> blocks before checking last line
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    lines = [l.strip() for l in t.splitlines() if l.strip()]
    for line in reversed(lines):
        ll = line.lower().strip(".,!?:; \"'*`#")
        if ll == "correct":
            return "Correct"
        if ll == "incorrect":
            return "Incorrect"
    for line in reversed(lines):
        ll = line.lower()
        if "incorrect" in ll:
            return "Incorrect"
        if "correct" in ll:
            return "Correct"
    return "Incorrect"


# ─────────────────────────────────────────────────────────────
# Calendar diff helpers
# ─────────────────────────────────────────────────────────────
DAY_RE = re.compile(r"^([A-Z][a-z]+):\s*$")
EVENT_RE = re.compile(r"^\s*(\d{2}:\d{2})-(\d{2}:\d{2})\s+(.+?)\s*(\[.*\])?\s*$")


def parse_state(text: str) -> dict[str, set[tuple]]:
    """Parse a `format_day_state_text` block into {day: {(start, end, summary, attendees), ...}}"""
    by_day: dict[str, set[tuple]] = {}
    cur = None
    for line in (text or "").splitlines():
        m = DAY_RE.match(line)
        if m:
            cur = m.group(1)
            by_day.setdefault(cur, set())
            continue
        if cur is None:
            continue
        em = EVENT_RE.match(line)
        if em:
            start, end, summary, attendees = em.groups()
            by_day[cur].add((start, end, summary.strip(), attendees or ""))
    return by_day


def diff_states(before_text: str, after_text: str) -> str:
    """Return a +/-/~ diff of before vs after states, day by day."""
    a = parse_state(before_text)
    b = parse_state(after_text)
    days = sorted(set(a) | set(b))
    out = []
    for d in days:
        ae = a.get(d, set())
        be = b.get(d, set())
        added = sorted(be - ae)
        removed = sorted(ae - be)
        # Detect "modified" pairs: same summary, different time
        added_by_sum = {x[2]: x for x in added}
        removed_by_sum = {x[2]: x for x in removed}
        common_sum = set(added_by_sum) & set(removed_by_sum)
        modified = []
        for s in common_sum:
            old = removed_by_sum[s]
            new = added_by_sum[s]
            modified.append((old, new))
            added.remove(new)
            removed.remove(old)
        if not (added or removed or modified):
            continue
        out.append(f"{d}:")
        for old, new in modified:
            out.append(f"  ~ {old[0]}-{old[1]} → {new[0]}-{new[1]}  {new[2]} {new[3]}".rstrip())
        for x in added:
            out.append(f"  + {x[0]}-{x[1]}  {x[2]} {x[3]}".rstrip())
        for x in removed:
            out.append(f"  - {x[0]}-{x[1]}  {x[2]} {x[3]}".rstrip())
    if not out:
        return "(no calendar changes)"
    return "\n".join(out)


# ─────────────────────────────────────────────────────────────
# Prompt variants
# ─────────────────────────────────────────────────────────────
EVAL_SYSTEM_BASE = """\
You evaluate a calendar assistant that has tools to search, create, update, \
and delete calendar events. Judge whether it completed the user's task.

First, determine which case applies:
1. The query has enough information for the agent to complete the task using \
its tools and the calendar data.
2. The query is ambiguous or incomplete — the agent cannot proceed without \
asking the user for clarification.

Then judge the agent's response accordingly. For case 1, check the BEFORE and \
AFTER calendar states — the state change is the ground truth. For case 2, the \
agent should have looked up candidates and asked the user to clarify.

Think step by step. Explain your reasoning in detail before giving a verdict.

On the very last line output exactly one word:
Correct
Incorrect
"""


def build_baseline(rec: dict) -> tuple[str, str, dict]:
    """Reproduce eval_judge_on_art prompt exactly."""
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    before = rec["before"]; after = rec["after"]
    user = (
        f"Query: {q}\n\n"
        f"Response: {final if final else '(no response)'}\n\n"
        f"Expected: {exp if exp else '(not specified)'}\n\n"
        f"Before:\n{before}\n\n"
        f"After:\n{after}\n\n"
        "Was the task completed correctly? End with one word: Correct or Incorrect."
    )
    return EVAL_SYSTEM_BASE, user, {}


def build_diff(rec: dict) -> tuple[str, str, dict]:
    """Show only the calendar diff instead of full Before+After dumps."""
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"Query: {q}\n\n"
        f"Response: {final if final else '(no response)'}\n\n"
        f"Expected: {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        "Was the task completed correctly? End with one word: Correct or Incorrect."
    )
    return EVAL_SYSTEM_BASE, user, {}


def build_diff_plus_full(rec: dict) -> tuple[str, str, dict]:
    """Diff + full state, in case diff alone loses context."""
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"Query: {q}\n\n"
        f"Response: {final if final else '(no response)'}\n\n"
        f"Expected: {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE:\n{rec['before']}\n\n"
        f"Full AFTER:\n{rec['after']}\n\n"
        "Was the task completed correctly? End with one word: Correct or Incorrect."
    )
    return EVAL_SYSTEM_BASE, user, {}


CHECKLIST_SYS = """\
You evaluate a calendar assistant. The assistant has tools to search, create, \
update, and delete calendar events.

Decide whether the assistant completed the task by working through this checklist:

(A) Did the user's query carry enough information to act, or was clarification \
    required? If clarification was required, the correct behavior is to look up \
    candidates and ask — not to silently guess.
(B) If action was required, did the calendar state actually change in the \
    expected way? Compare the calendar diff to the expected behavior. A claim \
    of success without a matching state change is INCORRECT.
(C) Did the assistant's text response answer the user's question? An assistant \
    that performs the right tool call but never returns the answer to the user \
    is INCORRECT.
(D) Did the assistant hallucinate events or attendees that aren't in the \
    calendar? That is INCORRECT.

Be lenient about cosmetic mismatches: date naming inconsistencies that are \
self-consistent with the actual stated weekday, attendee names vs emails when \
either uniquely identifies the person, and small wording differences in event \
summaries do NOT make the task incorrect.

Be strict about: wrong duration, wrong day, missing state change, fabricated \
events/attendees, or broken/garbled tool calls.

Think step by step through (A)-(D) before answering.

On the very last line output exactly one word:
Correct
Incorrect
"""


def build_cot_checklist(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"Query: {q}\n\n"
        f"Assistant's response: {final if final else '(no response)'}\n\n"
        f"Expected behavior: {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE:\n{rec['before']}\n\n"
        f"Full AFTER:\n{rec['after']}\n\n"
        "Work through (A)-(D) from the system prompt. End with Correct or Incorrect on the last line."
    )
    return CHECKLIST_SYS, user, {}


def build_cot_checklist_thinking(rec: dict) -> tuple[str, str, dict]:
    """Same as cot_checklist but enable Qwen3 /think for explicit reasoning."""
    sys, user, opts = build_cot_checklist(rec)
    opts = dict(opts)
    opts["enable_thinking"] = True
    opts["max_tokens"] = 2048
    return sys, user, opts


CHECKLIST_V2_SYS = """\
You evaluate a calendar assistant. The assistant has tools to search, create, \
update, and delete calendar events.

Decide whether the assistant completed the task by working through this checklist:

(A) Did the user's query carry enough information to act, or was clarification \
    required? If clarification was required, the correct behavior is either to \
    look up candidates and ask, OR to make a sensible default choice consistent \
    with a calendar agent's typical behavior. Asking is not strictly required \
    when a reasonable default exists.

(B) Did the calendar state actually change in the way the user wanted? \
    THE CALENDAR STATE IS THE GROUND TRUTH. Compare the BEFORE and AFTER. \
    If the user asked to schedule/move/cancel something and the calendar shows \
    the right change on the right weekday at the right time, the task is \
    correct — even if the assistant's TEXT response describes a different \
    calendar date (e.g. "Saturday, April 13" vs the actual Saturday in the \
    calendar). The state, not the text, decides.

(C) For pure information queries (no state change expected), did the response \
    include the answer the user actually wanted? Listing extra events alongside \
    the relevant one is fine — the user can read past it. A response that \
    OMITS the asked-for information, or that DENIES events that exist in the \
    calendar, is incorrect.

(D) Hallucination check: only call hallucination if you can verify an event \
    or attendee that the assistant mentions is NOT present in BEFORE or AFTER. \
    Do not claim hallucination just because the assistant says more than the \
    expected behavior listed.

(E) Hard failures (always Incorrect regardless of (A)-(D)): broken/garbled \
    tool calls in the response, claimed success without a matching state \
    change, wrong event was modified/deleted, or the agent flipped to a \
    completely different topic.

Be lenient about: cosmetic date naming, attendee names vs emails, breadth of \
listed events, asking for clarification on truly ambiguous queries. \
Be strict about: wrong duration when user specified one, hard failures from (E), \
denying events that exist in the calendar, fabricated events.

Think step by step through (A)-(E). On the very last line output exactly one \
word: Correct or Incorrect.
"""


def build_cot_checklist_v2(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"User query: {q}\n\n"
        f"Assistant's response (text shown to user): {final if final else '(no response)'}\n\n"
        f"Expected behavior: {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "Work through (A)-(E) from the system prompt. Remember: state is ground "
        "truth. End with Correct or Incorrect on the last line."
    )
    return CHECKLIST_V2_SYS, user, {}


def build_cot_checklist_v2_think(rec: dict) -> tuple[str, str, dict]:
    sys, user, opts = build_cot_checklist_v2(rec)
    opts = dict(opts)
    opts["enable_thinking"] = True
    opts["max_tokens"] = 2048
    return sys, user, opts


# ─────────────────────────────────────────────────────────────
# Per-category specialized prompts
# ─────────────────────────────────────────────────────────────
SCHEDULE_SYS = """\
You evaluate a calendar assistant's ability to schedule a new event.

THE CALENDAR STATE IS THE GROUND TRUTH. Ignore any date-naming inconsistencies \
in the assistant's TEXT response (e.g. it says "Saturday April 13" but the \
actual Saturday in the calendar is April 20). What matters is that the new \
event appears in BEFORE→AFTER on the correct weekday at the correct time.

CORRECT when ALL of these hold:
  • A new event with a matching summary appears in AFTER (and was not in BEFORE)
  • Day-of-week matches the user's request (Saturday, Wednesday, etc.)
  • Start time matches the user's request
  • Duration matches if the user specified one (1 hour, 2 hours, etc.)
  • Attendees specified by the user are present
  • If the user asked to schedule "later this week" or similar, any reasonable \
    day in range is acceptable

INCORRECT when:
  • No new event was created, or the wrong event was created
  • Wrong day-of-week (user said Wednesday but event landed on a different weekday)
  • Wrong start time
  • Wrong duration when user explicitly stated one
  • Specified attendees missing
  • Hard failures: garbled tool calls, agent flipped to a different topic

Think step by step. End with Correct or Incorrect on the last line.
"""

MODIFIER_SYS = """\
You evaluate a calendar assistant's ability to modify an existing event \
(reschedule, change time, cancel, change attendees, etc.).

The CALENDAR DIFF is your primary evidence. A claim of success without a \
matching diff is INCORRECT.

CORRECT when:
  • The right event was identified and modified per the user's request
  • The diff shows the requested change (move, delete, attendee change)
  • For cancel/delete: the event is gone in AFTER

INCORRECT when:
  • No state change but agent claimed success
  • Wrong event was modified (e.g. user asked about Thursday's dinner and \
    agent claims Saturday's was changed)
  • The change went the wrong direction (e.g. "push back 1 hour" but agent \
    moved earlier)
  • Garbled / malformed tool calls
  • A new duplicate event was created instead of modifying the existing one

Be lenient about: small wording differences, attendee email vs name, the \
agent claiming a different date than the actual day in the diff (the diff is \
ground truth).

Think step by step. End with Correct or Incorrect on the last line.
"""

IR_SYS = """\
You evaluate a calendar assistant's ability to answer an information-retrieval \
question (no state change expected).

CORRECT when:
  • The response contains the asked-for information accurately
  • The answer is consistent with the calendar state
  • Listing extra context alongside the asked-for answer is fine

INCORRECT when:
  • The response denies events that exist in the calendar (e.g. "no soccer \
    game on Saturday" when there is one)
  • The response gives wrong details (wrong attendees, wrong time)
  • The response is just a tool-call dump with no answer to the user
  • The response asks for clarification when the calendar clearly contains \
    the answer
  • The response hallucinates events/attendees not in the calendar

For attendees: the response can use either names or emails; if the user can \
identify who is attending, that's fine.

Think step by step. End with Correct or Incorrect on the last line.
"""

RELTIME_SYS = """\
You evaluate a calendar assistant's ability to answer a "today/tomorrow/yesterday/this week" \
question (no state change expected).

CORRECT when the response lists the events for the requested day(s), \
consistent with the calendar.

INCORRECT when:
  • Response says no events when there are events
  • Response lists events that aren't in the calendar (hallucination)
  • Wrong day was checked (e.g. user asked about yesterday but agent listed today)

Listing every event for the day is fine, even if the user phrased the \
question narrowly ("any meetings tomorrow?").

Think step by step. End with Correct or Incorrect on the last line.
"""

VAGUE_SYS = """\
You evaluate a calendar assistant on a vaguely-phrased question \
("what am I doing today", "fun this weekend", "meetings with X this week").

CORRECT when:
  • The response surfaces the relevant events (yoga as fun, meetings as \
    meetings, etc.). Including extra events is fine — the user can read past them.
  • For "what am I doing"-style questions, listing all events on the day is \
    explicitly correct.
  • For attendee-filtered questions, naming the right person/event counts as \
    correct even if attendee emails are not enumerated.

INCORRECT when:
  • Response denies events that exist
  • Response hallucinates events not in the calendar
  • Response asks for clarification when the calendar clearly contains the answer
  • Response is just a tool-call with no user-facing answer

Do NOT mark Incorrect just because the agent listed more events than the \
"expected behavior" enumerated, as long as the relevant ones are present.

Think step by step. End with Correct or Incorrect on the last line.
"""

CHAOS_SYS = """\
You evaluate a calendar assistant on a fragmentary or ambiguous query \
("Reschedule... seed...", "Pest control?", "Cancel that yoga thing").

For these queries TWO behaviors are both CORRECT:
  • Look up the most likely event and provide details / take a sensible action
  • Ask the user a focused clarifying question (after looking up candidates if \
    obvious from context)

INCORRECT when:
  • Agent denies events that exist in the calendar (e.g. fragment refers to a \
    real event but agent says "no events found")
  • Agent acts on a completely different topic than the fragment hinted at
  • Agent's response is garbled / malformed tool call
  • Agent claims success without state change (e.g. "rescheduled" but no diff)
  • For destructive actions (cancel/delete) without explicit user confirmation, \
    completing the action is still acceptable IF the right event was identified

Think step by step. End with Correct or Incorrect on the last line.
"""

COMPLEX_SYS = """\
You evaluate a calendar assistant on a multi-step request that combines \
several actions (e.g. "decline X and move Y to that slot", "cancel X and \
schedule Y instead", "is there a conflict if I schedule Z").

CORRECT when ALL requested actions show up in the state/response:
  • Each action requested → reflected in the diff or in the response
  • Conflict checks: response correctly reports whether the slot is free
  • Cancel-then-create: BOTH the cancel AND the create appear in the diff

INCORRECT when:
  • Any one of the requested actions was skipped or done incorrectly
  • Wrong event was identified (e.g. moved a different event than the user named)
  • Conflict reported wrong (says free but isn't, or says conflict but is free)
  • Created duplicate instead of moving

Think step by step through each requested action. End with Correct or \
Incorrect on the last line.
"""

CATEGORY_SYS_MAP = {
    "Schedule a Single Event": SCHEDULE_SYS,
    "Modifier & Correction (Rescheduling/Updates)": MODIFIER_SYS,
    "Information Retrieval (Querying)": IR_SYS,
    "Relative Time References (today, tomorrow, yesterday, this week)": RELTIME_SYS,
    "Vague & Contextual (Reasoning Required)": VAGUE_SYS,
    "Human Chaos (Edge Cases/Fragments)": CHAOS_SYS,
    "Complex Logic & Conflict (Advanced)": COMPLEX_SYS,
}


def build_per_category(rec: dict) -> tuple[str, str, dict]:
    """Route to the right specialized system prompt by category."""
    sys_prompt = CATEGORY_SYS_MAP.get(rec["cat"], CHECKLIST_V2_SYS)
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"User query: {q}\n\n"
        f"Assistant's response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "Apply the rules from the system prompt. End with Correct or Incorrect on the last line."
    )
    return sys_prompt, user, {}


def build_per_category_think(rec: dict) -> tuple[str, str, dict]:
    sys, user, opts = build_per_category(rec)
    opts = dict(opts); opts["enable_thinking"] = True; opts["max_tokens"] = 2048
    return sys, user, opts


CHECKLIST_V3_SYS = """\
You evaluate a calendar assistant. The assistant has tools to search, create, \
update, and delete calendar events.

THE CALENDAR STATE (BEFORE→AFTER) IS THE GROUND TRUTH. The assistant's text \
is just a description of what it did; the diff is what actually happened.

Decide Correct or Incorrect using these rules.

CORRECT requires ALL of:
  (1) The user's request was carried out, OR clarification was reasonably \
      requested for a truly ambiguous query.
  (2) For action requests (schedule/move/cancel), the diff matches the user's \
      intent on the correct weekday and at the correct time.
  (3) For info requests, the response includes the asked-for facts and does \
      not deny events that exist.
  (4) The response contains a user-facing answer (not just a tool call dump).

HARD INCORRECT — these always make it Incorrect, regardless of anything else:
  (E1) Diff is EMPTY but the assistant claims to have created/modified/deleted \
       something. Empty diff + success claim = Incorrect.
  (E2) The new event's DURATION does not match the user's stated duration. \
       "1 hour" → must be 60 minutes. "2 hours" → 120 minutes. A 30-minute \
       slot when user said 1 hour is Incorrect.
  (E3) The response is just a `<tool_call>` block with no user-facing prose \
       answering the question. Garbled / malformed tool calls are Incorrect.
  (E4) The agent denied an event that clearly exists in the BEFORE/AFTER state \
       (e.g. "no events on Saturday" when calendar shows events).
  (E5) The agent fabricated events/attendees not present in BEFORE or AFTER.
  (E6) The agent acted on the wrong event (e.g. user said Thursday's dinner, \
       agent modified Saturday's).
  (E7) For "move/push back by N hours/minutes", the diff went the wrong \
       direction (user said push back = later; if diff moved earlier, Incorrect).
  (E8) Multi-step requests where one step was skipped (e.g. "decline X and \
       move Y" where only Y was moved).

LENIENT — these alone do NOT make it Incorrect:
  (L1) The assistant's TEXT names a different calendar date (e.g. "Saturday \
       April 13") than the actual day in the diff — the state is ground truth.
  (L2) Listing extra events alongside the asked-for one for a vague query.
  (L3) Using attendee names instead of full emails.
  (L4) Creating an event on day X when user said "this week / later this week" \
       and any reasonable day works.
  (L5) Cosmetic wording differences in event summaries.
  (L6) A different ckpt/run of the same scenario where the diff is correct \
       even if the response text reads slightly differently.

For info-only queries with no expected state change, an empty diff is the \
NORMAL case, not a failure (E1 doesn't apply).

Think step by step through (1)-(4), then check (E1)-(E8) against the actual \
state, then check (L1)-(L6) before being strict. End with Correct or \
Incorrect on the very last line.
"""


def build_cot_checklist_v3(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "Apply the (1)-(4) rules, then (E1)-(E8) hard-incorrect checks, then "
        "(L1)-(L6) leniency. End with Correct or Incorrect on the last line."
    )
    return CHECKLIST_V3_SYS, user, {}


# Per-category v2: v2 base + a category-specific addendum injected as user prefix
CATEGORY_ADDENDA = {
    "Schedule a Single Event": (
        "CATEGORY HINT: This is a 'Schedule a Single Event' query. The user "
        "wants ONE new event added. (E1) and (E2) are the most common failure "
        "modes. Apply (L1) — ignore date naming, the day-of-week in the diff "
        "is what matters."
    ),
    "Modifier & Correction (Rescheduling/Updates)": (
        "CATEGORY HINT: This is a modifier/update query. Check the ~ line in "
        "the diff. (E1) and (E6) are the most common failures. The assistant "
        "may also accidentally CREATE a new event instead of UPDATING — that "
        "is Incorrect (look for + lines that should have been ~)."
    ),
    "Information Retrieval (Querying)": (
        "CATEGORY HINT: This is an information-retrieval query. The diff "
        "should be empty (no state change expected). (E3) and (E4) are the "
        "most common failures. Listing extra events past the asked-for one "
        "is fine (L2)."
    ),
    "Relative Time References (today, tomorrow, yesterday, this week)": (
        "CATEGORY HINT: This is a relative-time info query. Diff should be "
        "empty. The response must list events for the requested day(s). "
        "(E5) hallucinated events is the main risk."
    ),
    "Vague & Contextual (Reasoning Required)": (
        "CATEGORY HINT: This is a vague/contextual query. Listing extra "
        "events alongside the relevant one is acceptable (L2). Be lenient "
        "about scope mismatch (e.g. user said 'meetings' and agent listed "
        "all events including the meetings — that is Correct as long as the "
        "relevant items are there)."
    ),
    "Human Chaos (Edge Cases/Fragments)": (
        "CATEGORY HINT: This is a fragmentary/chaotic query. Both 'act with "
        "a sensible default' AND 'ask for clarification' are acceptable. "
        "Failures are: agent denied an event that exists, garbled tool call, "
        "or claimed success without state change."
    ),
    "Complex Logic & Conflict (Advanced)": (
        "CATEGORY HINT: This is a multi-step / conflict-aware query. ALL "
        "requested actions must show up in the diff or the response. (E8) "
        "skipped step is the most common failure. For 'cancel X and create "
        "Y instead', BOTH a removal AND an addition must appear in the diff."
    ),
}


def build_per_category_v3(rec: dict) -> tuple[str, str, dict]:
    """v3 base prompt + category-specific user-prefix hint."""
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    addendum = CATEGORY_ADDENDA.get(rec["cat"], "")
    user = (
        (addendum + "\n\n" if addendum else "") +
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "Apply (1)-(4), then (E1)-(E8), then (L1)-(L6). End with Correct or "
        "Incorrect on the last line."
    )
    return CHECKLIST_V3_SYS, user, {}


FEWSHOT_EXAMPLES = """\
Here are seven worked examples covering the common edge cases. Use them as \
calibration before judging the actual case below.

=== Example 1 (Schedule, date-naming inconsistency, CORRECT) ===
User: "Book me a haircut appointment for Saturday at 10 AM."
Response: "Scheduled for Saturday, April 13, from 10:00 AM to 10:30 AM."
Diff:
  Saturday:
    + 10:00-10:30  Haircut Appointment
Reasoning: The agent's text says "April 13" but the actual Saturday in this \
calendar is April 20. THE CALENDAR IS GROUND TRUTH. The event was added on \
Saturday at 10 AM as requested. The text date is just labelling.
Verdict: Correct

=== Example 2 (Schedule, duration mismatch, INCORRECT) ===
User: "Book a doctor's appointment for next Tuesday at 3:30 PM for one hour."
Response: "Doctor's appointment booked for Tuesday at 3:30 PM."
Diff:
  Tuesday:
    + 15:30-16:00  Doctor's Appointment
Reasoning: User explicitly said "one hour" but the event is only 30 minutes \
(15:30-16:00). Duration mismatch when user specified one is a hard failure.
Verdict: Incorrect

=== Example 3 (Modifier, no state change but claimed success, INCORRECT) ===
User: "Reschedule the client meeting on Thursday to 2 PM."
Response: "The client meeting has been rescheduled to Thursday, December 18, at 2 PM."
Diff: (no calendar changes)
Reasoning: Agent claimed success but the calendar didn't change. Empty diff \
+ success claim = Incorrect.
Verdict: Incorrect

=== Example 4 (Vague, over-broad listing acceptable, CORRECT) ===
User: "What meetings do I have on Wednesday?"
Response: "Hospital Rounds 8-12, Lunch with colleague 12:30-13:30, \
Vaccine Hesitancy Strategies 14-15, Family Time - Board Games 18-19."
Diff: (no calendar changes)
Reasoning: User asked for meetings; agent listed all events including a few \
non-meetings (lunch, family time). The actual meetings are correctly in the \
response. Extra context is fine.
Verdict: Correct

=== Example 5 (Chaos, sensible default action, CORRECT) ===
User: "Cancel that yoga thing"
Response: "The yoga class has been cancelled."
Diff:
  Tuesday:
    - 18:30-19:30  Yoga Class
Reasoning: Fragmentary query but unambiguous given calendar context. Agent \
identified the right event and removed it. Either acting OR asking for \
confirmation is acceptable on Chaos queries.
Verdict: Correct

=== Example 6 (Relative-time, hallucinated events, INCORRECT) ===
User: "Did I have any meetings yesterday?"
Response: "Yesterday you had: Brunch with Family 10-12, Visit to Park 14-16, \
Dinner with Spouse 19-21."
BEFORE state on yesterday: (only Volunteer 9-11 and Family Time 14-17 listed)
Diff: (no calendar changes)
Reasoning: The events the agent listed are NOT in the BEFORE state for that \
day. This is hallucination.
Verdict: Incorrect

=== Example 7 (Complex, skipped step, INCORRECT) ===
User: "Decline movie night and move my weekly meeting to that slot."
Response: "Movie night declined and weekly meeting moved."
Diff:
  Friday:
    ~ 17:00-18:00 → 18:30-19:30  Weekly Meeting
BEFORE/AFTER both still contain "Movie Night" on Friday at 18:30-19:30.
Reasoning: The meeting was moved correctly, but Movie Night was NOT actually \
declined (still in AFTER state). Multi-step request with one step skipped.
Verdict: Incorrect

Now judge the ACTUAL case below using the same approach.
"""


def build_fewshot(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    sys_prompt = CHECKLIST_V2_SYS + "\n\n" + FEWSHOT_EXAMPLES
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "End with Correct or Incorrect on the last line."
    )
    return sys_prompt, user, {"max_tokens": 1024}


def build_fewshot_per_category(rec: dict) -> tuple[str, str, dict]:
    """Few-shot examples + a single category-hint prefix."""
    sys_prompt, user, opts = build_fewshot(rec)
    addendum = CATEGORY_ADDENDA.get(rec["cat"], "")
    if addendum:
        user = addendum + "\n\n" + user
    return sys_prompt, user, opts


# Extended few-shot adding Chaos-specific failure modes (v2 of fewshot)
FEWSHOT_EXAMPLES_V2 = FEWSHOT_EXAMPLES + """
=== Example 8 (Chaos, agent gave up despite obvious context, INCORRECT) ===
User: "Oyster beds... Monday"
Response: "How can I assist you with oyster beds? Are you looking to \
schedule an event, add a reminder, or something else?"
BEFORE state on Monday includes: "07:00-08:00  Oyster Bed Check & Maintenance"
Diff: (no calendar changes)
Reasoning: The fragment plus the calendar context unambiguously refers to the \
existing event. Agent should have surfaced "Oyster Bed Check & Maintenance \
on Monday 7-8 AM" instead of asking what the user wants. Treating an \
obvious-from-context query as if there were no clue available = Incorrect.
Verdict: Incorrect

=== Example 9 (Chaos, agent took the wrong action, INCORRECT) ===
User: "Move my yoga."
Response: "The yoga class event has been successfully deleted from your calendar."
Diff:
  Tuesday:
    - 18:30-19:30  Yoga Class
Reasoning: User asked to MOVE; agent DELETED. Wrong action. The diff confirms \
the event was removed, not rescheduled.
Verdict: Incorrect

=== Example 10 (Chaos, vague non-answer that doesn't actually inform, INCORRECT) ===
User: "Dinner... Friday?"
Response: "Dinner is scheduled for Friday. If you'd like to make any changes \
or need further assistance, let me know!"
BEFORE state Friday: "18:00-20:00  Dinner with Fishing Buddies [...]"
Diff: (no calendar changes)
Reasoning: The response confirms a dinner exists but doesn't surface what or \
when. User wanted information; this is a non-answer. Helpful info-retrieval \
needs the actual event details.
Verdict: Incorrect

=== Example 11 (Chaos, look-up + clarify-on-action, CORRECT) ===
User: "Reschedule structural analysis"
Response: "The event 'Structural Analysis: Review Overpass Inspection Data' \
is scheduled for Tuesday, October 10, from 14:00 to 16:00. Would you like to \
reschedule this event? Please provide the new start and end times."
Diff: (no calendar changes)
Reasoning: Textbook Chaos-correct: agent identified the unique matching event \
and surfaced it, then asked the user to specify the new time. Both halves \
matter — looked up the event AND asked for the missing parameter.
Verdict: Correct
"""


def build_fewshot_v2(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    sys_prompt = CHECKLIST_V2_SYS + "\n\n" + FEWSHOT_EXAMPLES_V2
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "End with Correct or Incorrect on the last line."
    )
    return sys_prompt, user, {"max_tokens": 1024}


VARIANTS = {
    "baseline":               build_baseline,
    "diff":                   build_diff,
    "diff_plus_full":         build_diff_plus_full,
    "cot_checklist":          build_cot_checklist,
    "cot_checklist_think":    build_cot_checklist_thinking,
    "cot_checklist_v2":       build_cot_checklist_v2,
    "cot_checklist_v2_think": build_cot_checklist_v2_think,
    "per_category":           build_per_category,
    "per_category_think":     build_per_category_think,
    "cot_checklist_v3":       build_cot_checklist_v3,
    "per_category_v3":        build_per_category_v3,
    "fewshot":                build_fewshot,
    "fewshot_per_category":   build_fewshot_per_category,
    "fewshot_v2":             build_fewshot_v2,
    "fewshot_no_expected":    None,   # built below
    "fewshot_self_consistency": None, # built below
}


def build_fewshot_no_expected(rec: dict) -> tuple[str, str, dict]:
    """Same as fewshot_v2 but hide 'expected' (production setting)."""
    q = rec["query"]; final = rec["final"]
    diff = diff_states(rec["before"], rec["after"])
    sys_prompt = CHECKLIST_V2_SYS + "\n\n" + FEWSHOT_EXAMPLES_V2
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "End with Correct or Incorrect on the last line."
    )
    return sys_prompt, user, {"max_tokens": 1024}


def build_fewshot_self_consistency(rec: dict) -> tuple[str, str, dict]:
    """fewshot_v2 sampled 5x at T=0.7, majority vote."""
    sys_prompt, user, _ = build_fewshot_v2(rec)
    return sys_prompt, user, {"n_samples": 5, "temperature": 0.7, "max_tokens": 1024}


VARIANTS["fewshot_no_expected"] = build_fewshot_no_expected
VARIANTS["fewshot_self_consistency"] = build_fewshot_self_consistency


# fewshot_v3: add an info-retrieval example showing that an event surfaced
# from BEFORE state is NOT hallucination (judge was getting this confused).
FEWSHOT_EXAMPLES_V3 = FEWSHOT_EXAMPLES_V2 + """
=== Example 12 (Vague info-retrieval, surfaced event is NOT hallucination, CORRECT) ===
User: "When am I meeting with the engineer this week?"
Response: "You have a meeting with the engineer on Thursday from 15:00 to \
16:00. The event is titled 'Discuss automation project with plant engineer.'"
BEFORE state Thursday includes: \
"15:00-16:00  Discuss automation project with plant engineer  [plantengineer@plant.com]"
Diff: (no calendar changes — pure info query)
Reasoning: The event IS in the BEFORE state. The agent correctly identified \
and surfaced it. Empty diff is normal for info queries. Do NOT call \
hallucination when the event you're checking against is right there in BEFORE.
Verdict: Correct

=== Example 13 (Vague filter, listed all events incl. asked-for, CORRECT) ===
User: "What am I doing with Maria and David this week?"
Response: "Looking at the events for this week, you have planned activities \
with Maria and David. Here are the relevant events: 1. 'Dinner with Friends \
from Culinary School' on Monday from 18:30 to 20:00. 2. 'Brunch with Family' \
on Saturday from 13:00 to 15:00."
Reasoning: The agent surfaced relevant events. Even if the calendar lists \
Maria/David as attendees on different events, listing the ones present is \
acceptable. Don't penalize for the framing.
Verdict: Correct
"""


def build_fewshot_v3(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    sys_prompt = CHECKLIST_V2_SYS + "\n\n" + FEWSHOT_EXAMPLES_V3
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "End with Correct or Incorrect on the last line."
    )
    return sys_prompt, user, {"max_tokens": 1024}


VARIANTS["fewshot_v3"] = build_fewshot_v3


# v4: address specific misses found in v3 (RelTime denial / tool-call only,
# Complex partial-execution, Schedule duration when user says "next Tuesday").
FEWSHOT_EXAMPLES_V4 = FEWSHOT_EXAMPLES_V3 + """
=== Example 14 (Relative-time, denying events that exist, INCORRECT) ===
User: "What did I do yesterday?"
Response: "It seems you didn't have any calendar events yesterday."
BEFORE state for yesterday includes 4 events.
Diff: (no calendar changes — info query)
Reasoning: Agent denied events that ARE listed in the BEFORE state for the \
relevant day. This is a denial-of-existing-events failure. Doesn't matter \
that no state change is needed; the answer to the user is wrong.
Verdict: Incorrect

=== Example 15 (Relative-time, tool-call-only with no answer, INCORRECT) ===
User: "What is on my schedule tomorrow?"
Response (FULL): "<tool_call>\n{\"name\": \"list_events\", \"arguments\": {...}}\n</tool_call>"
Reasoning: The agent emitted only a tool-call block with no user-facing prose \
answering the question. The user reads this as no answer. (E3) Hard fail.
Verdict: Incorrect

=== Example 16 (Complex, both actions executed but wrong time, INCORRECT) ===
User: "I can't make the deposition on Thursday. Delete it and schedule a call \
with the therapist instead."
(User implicitly wanted the call at the deposition's time, e.g. 3:30 PM.)
Response: "The deposition has been deleted and the call with the therapist \
has been scheduled for Thursday from 10:30 AM to 12:00 PM."
Diff:
  Thursday:
    - 10:30-12:00  Deposition - Expert Witness
    + 10:30-12:00  Call with Therapist
Reasoning: Both actions happened, but the call landed at 10:30 AM (the \
deposition's old slot) rather than the user-implied 3:30 PM time. Substituting \
the wrong time on a multi-step request still counts as Incorrect — partial \
execution. Mark Correct ONLY if the timing was reasonable given the user's \
request.
Verdict: Incorrect

=== Example 17 (Schedule "next X day" — accept any next-X, CORRECT) ===
User: "Book a doctor's appointment for me next Tuesday at 3:30 PM."
Response: "Your doctor's appointment has been scheduled for next Tuesday at 3:30 PM."
Diff:
  Tuesday:
    + 15:30-16:00  Doctor's Appointment
Reasoning: The user said "next Tuesday at 3:30 PM" without specifying \
duration. The event was added on Tuesday at 15:30. WITHOUT a stated duration, \
30 minutes is acceptable. Don't penalize duration unless the user EXPLICITLY \
stated one (e.g. "for one hour", "for 2 hours", "lasting 30 minutes").
Verdict: Correct
"""


def build_fewshot_v4(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    sys_prompt = CHECKLIST_V2_SYS + "\n\n" + FEWSHOT_EXAMPLES_V4
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "End with Correct or Incorrect on the last line."
    )
    return sys_prompt, user, {"max_tokens": 1024}


VARIANTS["fewshot_v4"] = build_fewshot_v4


DAY_KEYWORDS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


def filter_state_to_days(text: str, days: list[str]) -> str:
    """Keep only the day-blocks listed; preserve all if days is empty."""
    if not days: return text
    keep = set(days)
    out = []
    cur_day = None; cur_lines = []
    for line in text.splitlines():
        m = DAY_RE.match(line)
        if m:
            if cur_day in keep:
                out.append(f"{cur_day}:")
                out.extend(cur_lines)
            cur_day = m.group(1); cur_lines = []
        elif cur_day is not None:
            cur_lines.append(line)
    if cur_day in keep:
        out.append(f"{cur_day}:")
        out.extend(cur_lines)
    return "\n".join(out) if out else "(no events on the addressed days)"


def days_in_text(text: str) -> list[str]:
    """Detect day-of-week keywords mentioned in a string (rough heuristic)."""
    found = []
    for d in DAY_KEYWORDS:
        if d.lower() in (text or "").lower():
            found.append(d)
    # Generic time refs that could span the week → keep all
    if any(w in (text or "").lower() for w in ["this week", "next week", "tomorrow", "yesterday", "today", "weekend"]):
        return DAY_KEYWORDS  # keep all
    return found if found else DAY_KEYWORDS  # fallback: all


def build_fewshot_v4_dayfocus(rec: dict) -> tuple[str, str, dict]:
    """v4 examples + state filtered to query-relevant days only."""
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    days = days_in_text(q + " " + (exp or ""))
    before_f = filter_state_to_days(rec["before"], days)
    after_f = filter_state_to_days(rec["after"], days)
    sys_prompt = CHECKLIST_V2_SYS + "\n\n" + FEWSHOT_EXAMPLES_V4
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"BEFORE state (days {', '.join(days[:3])}{'...' if len(days)>3 else ''}):\n{before_f}\n\n"
        f"AFTER state:\n{after_f}\n\n"
        "End with Correct or Incorrect on the last line."
    )
    return sys_prompt, user, {"max_tokens": 1024}


VARIANTS["fewshot_v4_dayfocus"] = build_fewshot_v4_dayfocus


def build_fewshot_v3_dayfocus(rec: dict) -> tuple[str, str, dict]:
    """v3 examples (the winning set) with state filtered to query-relevant days."""
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    days = days_in_text(q + " " + (exp or ""))
    before_f = filter_state_to_days(rec["before"], days)
    after_f = filter_state_to_days(rec["after"], days)
    sys_prompt = CHECKLIST_V2_SYS + "\n\n" + FEWSHOT_EXAMPLES_V3
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"BEFORE state (relevant days):\n{before_f}\n\n"
        f"AFTER state:\n{after_f}\n\n"
        "End with Correct or Incorrect on the last line."
    )
    return sys_prompt, user, {"max_tokens": 1024}


VARIANTS["fewshot_v3_dayfocus"] = build_fewshot_v3_dayfocus


# Route each trajectory to the variant that scored best on its category in
# prior runs (per-category bests from summary.csv, 2026-04-30).
ROUTER_MAP = {
    "Complex Logic & Conflict (Advanced)":     build_fewshot,           # 92.86
    "Human Chaos (Edge Cases/Fragments)":      build_fewshot_v3,        # 95.65
    "Information Retrieval (Querying)":        build_fewshot_v3,        # 97.62
    "Modifier & Correction (Rescheduling/Updates)": build_fewshot_v3,   # 97.56
    "Relative Time References (today, tomorrow, yesterday, this week)": build_fewshot,  # 94.74
    "Schedule a Single Event":                 build_fewshot_v3,        # 86.84
    "Vague & Contextual (Reasoning Required)": build_fewshot_v4_dayfocus, # 97.37
}


def build_router(rec: dict) -> tuple[str, str, dict]:
    """Dispatch to the variant that scored best on this category historically."""
    builder = ROUTER_MAP.get(rec["cat"], build_fewshot_v3)
    return builder(rec)


VARIANTS["router"] = build_router


# Verdict-only: same examples + rules but tell the model to output ONLY the
# verdict word, no reasoning. Tests "can we get accuracy with no CoT at
# inference time?" — this is what a trained judge would do.
VERDICT_ONLY_SUFFIX = "\n\nIMPORTANT: Do NOT explain your reasoning. Output ONLY one word: 'Correct' or 'Incorrect'. No other text."

def build_router_verdict_only(rec: dict) -> tuple[str, str, dict]:
    sys_, user, opts = build_router(rec)
    sys_ = sys_ + VERDICT_ONLY_SUFFIX
    opts = dict(opts); opts["max_tokens"] = 10
    return sys_, user, opts


VARIANTS["router_verdict_only"] = build_router_verdict_only


# ─────────────────────────────────────────────────────────────
# Fully bespoke per-category prompts: each category gets a focused system
# prompt with rules + worked examples drawn ONLY from that category.
# ─────────────────────────────────────────────────────────────

CAT_PROMPT_SCHEDULE = """\
You evaluate a calendar assistant on a SCHEDULE-A-NEW-EVENT query. The user \
asked to create one new event.

THE CALENDAR DIFF IS GROUND TRUTH. The assistant's text is just a description.

CORRECT when the diff shows a + (added) line that matches:
  - Day-of-week (the user's stated weekday)
  - Start time (the user's stated time, ±15 min only if user was vague)
  - Duration (only enforced if the user said "for X hours/minutes" explicitly)
  - Attendees (if the user named specific people/emails)
  - Summary roughly matching the user's words

INCORRECT when:
  - Diff is empty (no + line) — the agent claimed success but didn't act
  - Wrong weekday added
  - Wrong time
  - User said a duration and the new event has a different one
  - Garbled tool call instead of an answer

LENIENT (these alone don't make it Incorrect):
  - The agent's text mentions a calendar date that doesn't match the actual \
    weekday — the diff weekday is what counts
  - Different month/year in the agent's text vs the actual calendar
  - 30 min default when user said no duration

=== Example A (date-naming inconsistent but state right, CORRECT) ===
User: "Book me a haircut appointment for Saturday at 10 AM."
Response: "Scheduled for Saturday, April 13, from 10:00 to 10:30."
Diff:
  Saturday:
    + 10:00-10:30  Haircut Appointment
Verdict: Correct (state on Saturday is right; ignore the date label)

=== Example B (claimed success, no diff, INCORRECT) ===
User: "Book a dentist appointment for Friday at 10 AM."
Response: "Your dentist appointment has been successfully booked for Friday, December 15, from 10:00 to 11:00."
Diff: (no calendar changes)
Verdict: Incorrect (empty diff)

=== Example C (duration mismatch, INCORRECT) ===
User: "Book a doctor's appointment for next Tuesday at 3:30 PM for one hour."
Response: "Scheduled for Tuesday at 3:30 PM."
Diff:
  Tuesday:
    + 15:30-16:00  Doctor's Appointment
Verdict: Incorrect (user said one hour, agent created 30 min)

End with Correct or Incorrect on the very last line.
"""

CAT_PROMPT_MODIFIER = """\
You evaluate a calendar assistant on a MODIFY-EXISTING-EVENT query (move, \
reschedule, cancel, change attendees).

THE CALENDAR DIFF IS GROUND TRUTH.

CORRECT when:
  - For move/reschedule: diff shows a ~ line for the right event with the \
    new time matching what the user asked
  - For cancel/delete: diff shows a - line removing the right event
  - For attendee change: diff shows the right attendee added/removed

INCORRECT when:
  - Diff empty but agent claimed success
  - Wrong event was modified (look at the summary in the diff vs user's request)
  - For "push back / delay by N", the diff went the WRONG direction (earlier \
    instead of later)
  - Agent CREATED a new event instead of MODIFYING the existing one (you'll \
    see + and the original ~ unchanged — that is Incorrect)

LENIENT: agent's text describes a different date than actual diff day.

=== Example A (clean update, CORRECT) ===
User: "Move my meeting with the pharmaceutical rep to 4 PM on Monday."
Diff:
  Monday:
    ~ 15:30-16:30 → 16:00-17:00  Meeting with Pharmaceutical Rep
Verdict: Correct

=== Example B (no state change, INCORRECT) ===
User: "Reschedule the client meeting on Thursday to 2 PM."
Response: "The client meeting has been rescheduled to Thursday at 2 PM."
Diff: (no calendar changes)
Verdict: Incorrect

=== Example C (wrong direction, INCORRECT) ===
User: "Move my client consultation on Monday back by one hour." (push back = later)
Diff:
  Monday:
    ~ 09:00-10:30 → 08:00-09:30  Client Consultation
Verdict: Incorrect (moved earlier instead of later)

=== Example D (created new instead of updating, INCORRECT) ===
User: "Add 30 more minutes to the Ventilation Problem on Thursday." (existing 8-10)
Diff:
  Thursday:
    + 10:00-10:30  Ventilation Problem
Verdict: Incorrect (existing event unchanged; new duplicate added)

End with Correct or Incorrect on the very last line.
"""

CAT_PROMPT_IR = """\
You evaluate a calendar assistant on an INFORMATION-RETRIEVAL query (the user \
wants to know something — no state change expected).

The calendar diff should typically be EMPTY for these. That is normal, not a \
failure.

CORRECT when:
  - The response contains the asked-for fact (time, attendees, location...)
  - The fact is consistent with what the BEFORE state shows
  - Listing extra context alongside the answer is fine

INCORRECT when:
  - Response denies events that ARE in the BEFORE state
  - Response has wrong details (wrong time, wrong attendees)
  - Response is just a tool-call block with no user-facing prose
  - Response fabricates events/attendees not present in the BEFORE state
  - Response asks for clarification when the calendar clearly contains the answer

For attendees: names OR emails are both acceptable.

=== Example A (clean answer, CORRECT) ===
User: "What time is the Yoga Class on Monday?"
BEFORE Monday includes: "18:30-19:30  Yoga Class"
Response: "The Yoga Class is scheduled from 18:30 to 19:30 on Monday."
Verdict: Correct

=== Example B (denied existing event, INCORRECT) ===
User: "What time is the soccer game on Saturday?"
BEFORE Saturday includes: "09:00-12:00  Attend Kids Soccer Game"
Response: "There are no events listed for Saturday."
Verdict: Incorrect

=== Example C (only a tool call, INCORRECT) ===
User: "What time is my book club meeting?"
Response: `<tool_call>{"name":"list_events","arguments":{...}}</tool_call>`
Verdict: Incorrect

End with Correct or Incorrect on the very last line.
"""

CAT_PROMPT_RELTIME = """\
You evaluate a calendar assistant on a RELATIVE-TIME query (today/tomorrow/\
yesterday/this week — no state change expected).

CORRECT when the response lists the events for the requested day(s), \
matching what is in the BEFORE state for that day.

INCORRECT when:
  - Response says "no events" but the BEFORE state has events on the \
    requested day
  - Response lists events that AREN'T in the BEFORE state for that day \
    (hallucination)
  - Response is just a tool call with no user-facing answer
  - Wrong day was checked

The diff should be empty for these (info query). Empty diff is fine.

Listing every event for the day is acceptable, even if the user phrased the \
question narrowly ("any meetings tomorrow?"). Extra context = fine.

=== Example A (clean, CORRECT) ===
User: "What's on tomorrow's agenda?"
BEFORE tomorrow has: 4 events listed
Response: lists those 4 events with times.
Verdict: Correct

=== Example B (denied real events, INCORRECT) ===
User: "What did I do yesterday?"
BEFORE yesterday has 4 events
Response: "You didn't have any events yesterday."
Verdict: Incorrect

=== Example C (hallucinated events, INCORRECT) ===
User: "Did I have any meetings yesterday?"
BEFORE yesterday: 2 events (Volunteer, Family Time)
Response: lists "Brunch with Family", "Park Visit", "Dinner Date" — not in \
the BEFORE state.
Verdict: Incorrect

=== Example D (only a tool call, INCORRECT) ===
User: "What is on my schedule tomorrow?"
Response: `<tool_call>{"name":"list_events","arguments":{...}}น้ำตา`
Verdict: Incorrect

End with Correct or Incorrect on the very last line.
"""

CAT_PROMPT_VAGUE = """\
You evaluate a calendar assistant on a VAGUE/CONTEXTUAL query like "what am \
I doing today", "fun this weekend", "meetings with X this week".

The diff should be EMPTY (no state change). Empty diff is fine.

CORRECT when:
  - The response surfaces the relevant events from the BEFORE state
  - For "what meetings"/"fun events"/"meetings with X" — listing extra events \
    alongside the relevant ones is FINE. The user can read past them.
  - Including the right items + extras = Correct

INCORRECT when:
  - Response denies events that exist in the BEFORE state
  - Response hallucinates events not in BEFORE
  - Response is a tool-call block with no user-facing answer
  - Response asks for clarification when the calendar clearly contains the \
    answer (the agent could just look it up)

DO NOT mark Incorrect for "listed too many events" or "scope was broader \
than asked". Surfacing the right answer is what counts.

=== Example A (broad listing including the answer, CORRECT) ===
User: "What meetings do I have on Wednesday?"
Response lists 4 events including the 2 actual meetings + 2 non-meetings.
Verdict: Correct (the meetings are present)

=== Example B (correct event surfaced, CORRECT) ===
User: "When am I meeting with the engineer this week?"
BEFORE Thursday: "15:00-16:00  Discuss automation project with plant engineer"
Response: "Thursday from 15:00 to 16:00."
Verdict: Correct (event IS in BEFORE — not hallucination)

=== Example C (denied existing, INCORRECT) ===
User: "What meetings do I have with plant supervisors?"
BEFORE: events tagged with plant supervisors as attendees exist
Response: "You have no meetings with plant supervisors."
Verdict: Incorrect

=== Example D (over-broad with all events, still CORRECT) ===
User: "What am I doing with Maria and David this week?"
Response: lists all events for the week.
Verdict: Correct (relevant ones included; extras are fine)

End with Correct or Incorrect on the very last line.
"""

CAT_PROMPT_CHAOS = """\
You evaluate a calendar assistant on a CHAOS / FRAGMENT query — short, \
ambiguous, missing words ("Reschedule... seed...", "Pest control?", \
"Cancel that yoga thing", "Oyster beds... Monday").

For these queries TWO behaviors are both acceptable:
  (a) Look up the most likely event and provide details OR take the action
  (b) Look up candidates first, then ask the user a focused clarifying \
      question (especially for destructive actions)

INCORRECT when:
  - Response denies events that exist in the calendar (e.g. fragment refers \
    to a real event but agent says "no events found")
  - Response is a generic "could you give more details?" when the calendar \
    context makes the query unambiguous
  - Response asks generically without surfacing what IS in the calendar
  - Response acts on a completely different topic than the fragment hinted at
  - Response is a garbled / malformed tool call
  - Response claims success but diff is empty
  - Response surfaces the event name but doesn't actually answer (vague \
    confirmation like "Dinner is scheduled for Friday" with no details)

=== Example A (took sensible action, CORRECT) ===
User: "Cancel that yoga thing"
Response: "The yoga class has been cancelled."
Diff:
  Tuesday: - 18:30-19:30  Yoga Class
Verdict: Correct

=== Example B (looked up + asked for action, CORRECT) ===
User: "Reschedule structural analysis"
Response: "The 'Structural Analysis: Review Overpass Inspection Data' is on \
Tuesday October 10, 14:00-16:00. What new time would you like?"
Verdict: Correct (looked up AND asked for missing param)

=== Example C (generic ask despite obvious context, INCORRECT) ===
User: "Oyster beds... Monday"
BEFORE Monday: "07:00-08:00  Oyster Bed Check & Maintenance"
Response: "How can I assist with oyster beds? Are you looking to schedule, \
add a reminder, or something else?"
Verdict: Incorrect (calendar makes it unambiguous; should have surfaced the event)

=== Example D (wrong action, INCORRECT) ===
User: "Move my yoga."
Response: "The yoga class event has been deleted."
Diff: - Yoga Class
Verdict: Incorrect (user said move, agent deleted)

=== Example E (vague non-answer, INCORRECT) ===
User: "Dinner... Friday?"
BEFORE Friday: "18:00-20:00  Dinner with Fishing Buddies"
Response: "Dinner is scheduled for Friday."
Verdict: Incorrect (didn't surface the actual details)

End with Correct or Incorrect on the very last line.
"""

CAT_PROMPT_COMPLEX = """\
You evaluate a calendar assistant on a COMPLEX / MULTI-STEP query that \
combines multiple actions (e.g. "decline X and move Y to that slot", \
"cancel X and schedule Y instead", "is there a conflict if I schedule Z").

ALL requested actions must be reflected in the diff or response.

CORRECT when:
  - Each action requested is reflected in the diff (cancel→deletion line, \
    create→addition line, move→modification line)
  - Conflict checks correctly report whether the slot is free given BEFORE
  - Cancel-then-create: BOTH the cancel AND the create appear in the diff

INCORRECT when:
  - Any one of the requested actions was skipped or done incorrectly
  - For "is there a conflict?" the answer denies an existing event
  - Created a duplicate instead of moving (+ line added but the original \
    untouched)
  - Multi-step task only half-done

=== Example A (cancel+create both done, CORRECT) ===
User: "I can't make the deposition on Thursday. Delete it and schedule a \
call with the therapist instead."
Diff:
  Thursday:
    - 10:30-12:00  Deposition - Expert Witness
    + 10:30-12:00  Call with Therapist
Verdict: Correct (both actions in diff)

=== Example B (skipped a step, INCORRECT) ===
User: "Decline movie night this Friday and move my weekly meeting to that slot."
Diff:
  Friday: ~ 17:00-18:00 → 18:30-19:30  Weekly Meeting
BEFORE/AFTER both still contain "Family Movie Night" on Friday at 18:30-19:30.
Verdict: Incorrect (movie night was not declined — only the meeting moved)

=== Example C (correct conflict report, CORRECT) ===
User: "I want to schedule a meeting Wednesday 7:30-9 PM, is there already a meeting?"
BEFORE Wednesday: nothing in 19:30-21:00
Response: "No meeting scheduled in that slot. You can proceed."
Verdict: Correct

=== Example D (denied existing event for conflict check, INCORRECT) ===
User: "Schedule meeting Wednesday 7:30-9 PM, is there already a meeting?"
BEFORE Wednesday 19:30-21:00: "Neighborhood meeting"
Response: "No meetings scheduled in that slot."
Verdict: Incorrect (denied an existing event)

End with Correct or Incorrect on the very last line.
"""


CAT_BESPOKE_PROMPTS = {
    "Schedule a Single Event": CAT_PROMPT_SCHEDULE,
    "Modifier & Correction (Rescheduling/Updates)": CAT_PROMPT_MODIFIER,
    "Information Retrieval (Querying)": CAT_PROMPT_IR,
    "Relative Time References (today, tomorrow, yesterday, this week)": CAT_PROMPT_RELTIME,
    "Vague & Contextual (Reasoning Required)": CAT_PROMPT_VAGUE,
    "Human Chaos (Edge Cases/Fragments)": CAT_PROMPT_CHAOS,
    "Complex Logic & Conflict (Advanced)": CAT_PROMPT_COMPLEX,
}


def build_cat_bespoke(rec: dict) -> tuple[str, str, dict]:
    """One bespoke system prompt + examples per category."""
    sys_prompt = CAT_BESPOKE_PROMPTS.get(rec["cat"], CHECKLIST_V2_SYS + "\n\n" + FEWSHOT_EXAMPLES_V3)
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff (+ added, - removed, ~ modified):\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        "End with Correct or Incorrect on the very last line."
    )
    return sys_prompt, user, {"max_tokens": 1024}


VARIANTS["cat_bespoke"] = build_cat_bespoke


# ─────────────────────────────────────────────────────────────
# Two-stage verifier: stage-1 verdict + reasoning is fed to a stage-2 verifier
# that decides FLIP or KEEP.
# ─────────────────────────────────────────────────────────────

VERIFIER_CONSERVATIVE_SYS = """\
You are auditing another judge's verdict on a calendar-assistant task.

CRITICAL CALIBRATION: the prior judge is correct 95% of the time. Your default \
answer is KEEP. Only output FLIP if you observe a HARD-RULE VIOLATION that the \
prior judge missed — i.e. clear, unambiguous evidence that the verdict is wrong.

HARD-RULE VIOLATIONS that justify FLIP (and ONLY these):
  (V1) Empty diff but agent claimed success on an action query (schedule/move/\
       cancel) AND the prior judge said Correct. → FLIP to Incorrect.
  (V2) Diff shows a `+` line for a NEW event when the user asked to MODIFY an \
       existing one (visible in BEFORE), AND prior said Correct. → FLIP.
  (V3) Multi-step request: user asked for action A AND action B, but only one \
       appears in the diff (the other is still unchanged in BEFORE/AFTER), AND \
       prior said Correct. → FLIP.
  (V4) Response is ONLY a `<tool_call>` block with no user-facing prose, AND \
       prior said Correct. → FLIP to Incorrect.
  (V5) Response denies an event that IS plainly in BEFORE state, AND prior said \
       Correct. → FLIP.
  (V6) "Push back / delay by N" but diff moved earlier, AND prior said Correct. \
       → FLIP.

DO NOT FLIP for:
  - Disagreements about cosmetic date naming
  - The prior judge being more lenient or more strict on borderline cases
  - The prior judge listing extra events in their reasoning
  - The prior judge using slightly different language than you would
  - Anything that requires you to "interpret" — only flip on clear violations

If you're not 100% sure a HARD-RULE VIOLATION exists, output KEEP.

End with exactly one of these on the very last line:
  KEEP
  FLIP
"""

VERIFIER_SYS = """\
You are reviewing a verdict given by another judge on whether a calendar \
assistant completed the user's task. Your job is to decide whether the prior \
judge's verdict is RIGHT or WRONG, given the same evidence.

You will see:
  • The original query
  • The agent's user-facing response
  • The expected behavior hint
  • The calendar diff and full BEFORE/AFTER state
  • The prior judge's reasoning + verdict

Apply these correction principles when reviewing:

(1) THE CALENDAR DIFF IS GROUND TRUTH for action queries (schedule/move/cancel).
    If the diff is empty but the agent claimed success → Incorrect.
    If the diff shows the right change on the right weekday → Correct (regardless
    of date-naming inconsistencies in the agent's text).

(2) For info queries (no state change expected): the diff should be empty.
    Empty diff is NOT a failure for these.
    Failure modes: denying events that exist in BEFORE, hallucinating events
    not in BEFORE, tool-call-only response with no prose.

(3) For "duplicate-instead-of-update" cases: if the user asked to MODIFY an
    existing event but the diff shows a `+` line (new event) without a
    matching `~` line (modification), that is INCORRECT — the agent created
    a duplicate instead of updating.

(4) For multi-step requests: ALL requested actions must be in the diff. If
    the user asked "decline X AND move Y" and the diff only shows Y moved
    (with X still present in BEFORE/AFTER), that is INCORRECT.

(5) For "push back / delay by N" requests: pushing back usually means LATER
    in time. If the diff moved earlier, that is INCORRECT.

(6) Don't be over-strict. Asking for clarification on truly ambiguous queries
    is acceptable. Listing extra context alongside the asked-for answer is
    acceptable. Cosmetic date naming differences don't matter.

Decide: did the prior judge get it right?

Output exactly one of these on the very last line:
  KEEP    — the prior verdict is correct
  FLIP    — the prior verdict is wrong, the opposite is correct
"""


_VERIFIER_MODE = {"sys": VERIFIER_SYS}  # toggleable


def build_verifier_prompt(rec: dict, stage1_verdict: str, stage1_reasoning: str) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec["expected"]
    diff = diff_states(rec["before"], rec["after"])
    user = (
        f"User query: {q}\n\n"
        f"Assistant's user-facing response: {final if final else '(no response)'}\n\n"
        f"Expected behavior (hint): {exp if exp else '(not specified)'}\n\n"
        f"Calendar diff:\n{diff}\n\n"
        f"Full BEFORE state:\n{rec['before']}\n\n"
        f"Full AFTER state:\n{rec['after']}\n\n"
        f"=== Prior judge's reasoning ===\n{stage1_reasoning[-1500:]}\n\n"
        f"=== Prior judge's verdict ===\n{stage1_verdict}\n\n"
        "Audit this verdict per the system rules. End with KEEP or FLIP on the last line."
    )
    return _VERIFIER_MODE["sys"], user, {"max_tokens": 1024}


def extract_keep_flip(text: str) -> str:
    """Last-line scan for KEEP / FLIP decision."""
    if not text: return "KEEP"
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    lines = [l.strip().strip(".,!?:; \"'*`#") for l in t.splitlines() if l.strip()]
    for line in reversed(lines):
        ll = line.lower()
        if ll == "flip":
            return "FLIP"
        if ll == "keep":
            return "KEEP"
    for line in reversed(lines):
        ll = line.lower()
        if "flip" in ll: return "FLIP"
        if "keep" in ll: return "KEEP"
    return "KEEP"


async def run_verifier_pass(stage1_results: list[dict], concurrency: int = 16) -> tuple[list[dict], dict]:
    """Run the verifier on each stage-1 result. Return updated results + flip stats."""
    sem = asyncio.Semaphore(concurrency)
    out = [None] * len(stage1_results)

    async with httpx.AsyncClient(base_url=API_BASE, timeout=300) as client:
        async def go(i, r):
            async with sem:
                rec = RECS[r["idx"]]
                stage1_verdict = r["verdict"]
                stage1_reasoning = r.get("raw", "")
                system, user, opts = build_verifier_prompt(rec, stage1_verdict, stage1_reasoning)
                ver = await query_one(client, r["idx"], system, user, opts)
                decision = extract_keep_flip(ver["raw"])
                final_verdict = stage1_verdict
                if decision == "FLIP":
                    final_verdict = "Incorrect" if stage1_verdict == "Correct" else "Correct"
                out[i] = {
                    **r,
                    "verifier_decision": decision,
                    "verifier_raw": ver["raw"],
                    "stage1_verdict": stage1_verdict,
                    "verdict": final_verdict,
                }
        await asyncio.gather(*(go(i, r) for i, r in enumerate(stage1_results)))

    flips = sum(1 for r in out if r["verifier_decision"] == "FLIP")
    return out, {"flipped": flips, "kept": len(out) - flips}


def build_verify_router(rec: dict) -> tuple[str, str, dict]:
    """Stage-1 builder for the verify path is the router."""
    return build_router(rec)


VARIANTS["verify_router"] = build_verify_router


# ─────────────────────────────────────────────────────────────
# Async query
# ─────────────────────────────────────────────────────────────
async def query_one(client: httpx.AsyncClient, idx: int, system: str, user: str, opts: dict) -> dict:
    n_samples = opts.get("n_samples", 1)
    temperature = opts.get("temperature", 0.0 if n_samples == 1 else 0.7)
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens": opts.get("max_tokens", 1024),
        "n": n_samples,
    }
    if opts.get("enable_thinking"):
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    t0 = time.time()
    for attempt in range(3):
        try:
            r = await client.post("/chat/completions", json=payload, timeout=300)
            r.raise_for_status()
            choices = r.json()["choices"]
            verdicts = [extract_verdict(c["message"]["content"]) for c in choices]
            # Majority vote (ties → Incorrect, since "no answer = fail")
            counts = {"Correct": verdicts.count("Correct"), "Incorrect": verdicts.count("Incorrect")}
            verdict = "Correct" if counts["Correct"] > counts["Incorrect"] else "Incorrect"
            content = choices[0]["message"]["content"]
            return {"idx": idx, "raw": content, "verdict": verdict, "all_verdicts": verdicts,
                    "latency_s": round(time.time()-t0, 2)}
        except Exception as e:
            if attempt == 2:
                return {"idx": idx, "raw": f"[ERROR {e}]", "verdict": "Incorrect", "latency_s": round(time.time()-t0, 2)}
            await asyncio.sleep(2)


async def run_variant(variant: str, concurrency: int, limit: int | None,
                      max_tokens_override: int | None = None,
                      variant_tag: str | None = None) -> dict:
    builder = VARIANTS[variant]
    items = list(enumerate(RECS))
    if limit:
        items = items[:limit]
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = [None] * len(items)
    per_call_latencies: list[float] = []

    async with httpx.AsyncClient(base_url=API_BASE, timeout=300) as client:
        async def go(i, rec):
            async with sem:
                system, user, opts = builder(rec)
                if max_tokens_override is not None:
                    opts = dict(opts); opts["max_tokens"] = max_tokens_override
                res = await query_one(client, i, system, user, opts)
                results[i] = res
                per_call_latencies.append(res["latency_s"])

        t0 = time.time()
        await asyncio.gather(*(go(i, rec) for i, rec in items))
        wall = time.time() - t0

    # Score
    correct = total = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [agree, total]
    confusion = defaultdict(int)
    for r in results:
        if r is None: continue
        i = r["idx"]
        truth = TRUTH[i]
        cat = RECS[i]["cat"]
        total += 1
        by_cat[cat][1] += 1
        if r["verdict"] == truth:
            correct += 1
            by_cat[cat][0] += 1
        confusion[(truth, r["verdict"])] += 1

    p50 = sorted(per_call_latencies)[len(per_call_latencies)//2] if per_call_latencies else 0
    p90 = sorted(per_call_latencies)[int(0.9*len(per_call_latencies))] if per_call_latencies else 0
    name = variant + (f"_{variant_tag}" if variant_tag else "")
    summary = {
        "variant": name,
        "n": total,
        "correct": correct,
        "acc_pct": round(100 * correct / total, 2),
        "wall_s": round(wall, 1),
        "p50_s": round(p50, 2),
        "p90_s": round(p90, 2),
        "concurrency": concurrency,
        "per_category": {c: round(100 * a / t, 2) for c, (a, t) in by_cat.items()},
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
    }
    # Save per-trajectory results
    with open(OUT_DIR / f"{name}.jsonl", "w") as f:
        for r in results:
            if r is None: continue
            f.write(json.dumps({**r, "truth": TRUTH[r["idx"]], "cat": RECS[r["idx"]]["cat"]}) + "\n")
    # Append summary row
    write_summary_row(summary)
    return summary


def write_summary_row(summary: dict):
    headers = ["timestamp", "variant", "n", "acc_pct", "wall_s", "p50_s", "p90_s",
               "Complex", "Chaos", "IR", "Modifier", "RelTime", "Schedule", "Vague",
               "C->C", "C->I", "I->C", "I->I"]
    cat_short = {
        "Complex Logic & Conflict (Advanced)": "Complex",
        "Human Chaos (Edge Cases/Fragments)": "Chaos",
        "Information Retrieval (Querying)": "IR",
        "Modifier & Correction (Rescheduling/Updates)": "Modifier",
        "Relative Time References (today, tomorrow, yesterday, this week)": "RelTime",
        "Schedule a Single Event": "Schedule",
        "Vague & Contextual (Reasoning Required)": "Vague",
    }
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "variant": summary["variant"],
        "n": summary["n"],
        "acc_pct": summary["acc_pct"],
        "wall_s": summary["wall_s"],
        "p50_s": summary.get("p50_s", ""),
        "p90_s": summary.get("p90_s", ""),
    }
    for full, short in cat_short.items():
        row[short] = summary["per_category"].get(full, "")
    for k in ["C->C", "C->I", "I->C", "I->I"]:
        truth, pred = k.split("->")
        truth_full = "Correct" if truth == "C" else "Incorrect"
        pred_full = "Correct" if pred == "C" else "Incorrect"
        row[k] = summary["confusion"].get(f"{truth_full}->{pred_full}", 0)
    write_header = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if write_header: w.writeheader()
        w.writerow(row)


async def run_variant_with_verifier(variant: str, concurrency: int, limit: int | None) -> dict:
    """Run stage-1 (variant), then stage-2 verifier; report final scored results."""
    builder = VARIANTS[variant]
    items = list(enumerate(RECS))
    if limit: items = items[:limit]
    sem = asyncio.Semaphore(concurrency)
    stage1: list[dict] = [None] * len(items)

    async with httpx.AsyncClient(base_url=API_BASE, timeout=300) as client:
        async def s1(i, rec):
            async with sem:
                system, user, opts = builder(rec)
                stage1[i] = await query_one(client, i, system, user, opts)
        t0 = time.time()
        await asyncio.gather(*(s1(i, rec) for i, rec in items))
        wall_s1 = time.time() - t0

    stage1_acc = sum(1 for r in stage1 if r and r["verdict"] == TRUTH[r["idx"]])
    print(f"  stage-1 done in {wall_s1:.1f}s; raw acc {100*stage1_acc/len(stage1):.2f}%")

    t0 = time.time()
    final_results, flip_stats = await run_verifier_pass(stage1, concurrency=concurrency)
    wall_s2 = time.time() - t0
    print(f"  verifier done in {wall_s2:.1f}s; flipped={flip_stats['flipped']}, kept={flip_stats['kept']}")

    # Score
    correct = total = 0
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion: dict = defaultdict(int)
    for r in final_results:
        if r is None: continue
        i = r["idx"]; truth = TRUTH[i]; cat = RECS[i]["cat"]
        total += 1
        by_cat[cat][1] += 1
        if r["verdict"] == truth:
            correct += 1
            by_cat[cat][0] += 1
        confusion[(truth, r["verdict"])] += 1

    summary = {
        "variant": f"{variant}+verify",
        "n": total,
        "correct": correct,
        "acc_pct": round(100 * correct / total, 2),
        "wall_s": round(wall_s1 + wall_s2, 1),
        "concurrency": concurrency,
        "per_category": {c: round(100 * a / t, 2) for c, (a, t) in by_cat.items()},
        "confusion": {f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
        "flips": flip_stats["flipped"],
    }
    # Save per-trajectory
    with open(OUT_DIR / f"{variant}_verify.jsonl", "w") as f:
        for r in final_results:
            if r is None: continue
            f.write(json.dumps({**r, "truth": TRUTH[r["idx"]], "cat": RECS[r["idx"]]["cat"]}) + "\n")
    write_summary_row(summary)

    # Verifier flip analysis
    flip_correct = flip_wrong = keep_correct = keep_wrong = 0
    for r in final_results:
        i = r["idx"]; truth = TRUTH[i]
        s1 = r["stage1_verdict"]; final = r["verdict"]
        if r["verifier_decision"] == "FLIP":
            if final == truth: flip_correct += 1   # flip helped
            else: flip_wrong += 1                  # flip hurt
        else:
            if final == truth: keep_correct += 1
            else: keep_wrong += 1
    print(f"\n  Verifier flip analysis:")
    print(f"    Flips that HELPED  (s1 wrong → s2 right): {flip_correct}")
    print(f"    Flips that HURT    (s1 right → s2 wrong): {flip_wrong}")
    print(f"    Keeps that HELPED  (s1 right → still right): {keep_correct}")
    print(f"    Keeps that MISSED  (s1 wrong → still wrong): {keep_wrong}")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default=None, help="prompt variant name; see --list")
    p.add_argument("--list", action="store_true", help="list available variants and exit")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--limit", type=int, default=None, help="run only first N trajectories (for quick testing)")
    p.add_argument("--verify", action="store_true", help="run a verifier pass after stage-1")
    p.add_argument("--verifier", choices=["default", "conservative"], default="default",
                   help="which verifier system prompt to use")
    p.add_argument("--max-tokens", type=int, default=None, help="override max_tokens for the run")
    p.add_argument("--variant-tag", default=None, help="suffix for variant name (results CSV)")
    args = p.parse_args()

    if args.list:
        print("Available variants:")
        for k in VARIANTS: print(f"  {k}")
        return
    if not args.variant:
        p.error("--variant required (or --list)")
    if args.variant not in VARIANTS:
        p.error(f"unknown variant {args.variant}; see --list")

    print(f"Variant: {args.variant}{' +verify('+args.verifier+')' if args.verify else ''} | n={args.limit or len(RECS)} | concurrency={args.concurrency}")
    print(f"API: {API_BASE}  model={MODEL}")
    if args.verify:
        if args.verifier == "conservative":
            _VERIFIER_MODE["sys"] = VERIFIER_CONSERVATIVE_SYS
        summary = asyncio.run(run_variant_with_verifier(args.variant, args.concurrency, args.limit))
    else:
        summary = asyncio.run(run_variant(args.variant, args.concurrency, args.limit,
                                          max_tokens_override=args.max_tokens,
                                          variant_tag=args.variant_tag))
    print(f"\nResult: {summary['acc_pct']}%  ({summary['correct']}/{summary['n']})  in {summary['wall_s']}s")
    print(f"Per-category: {summary['per_category']}")
    print(f"Confusion: {summary['confusion']}")
    print(f"\nFull summary appended to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
