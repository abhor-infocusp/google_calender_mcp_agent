"""Judge prompts + router (lifted verbatim from scripts/eval/judge_prompt_tune.py).

Canonical winning configuration: the `router` variant on Qwen3-14B fp8.
- 95.44% on the manual oracle (185 Correct / 100 Incorrect, 2026-05-01).
- Per-category dispatch (ROUTER_MAP) to one of three few-shot prompt variants.
- Reasoning is generated server-side; only the verdict is exposed to callers.

PROMPT_VERSION is stamped in every server response and JSONL log line so
mid-run prompt edits are detectable as reward-signal drift.
"""
from __future__ import annotations

import re

PROMPT_VERSION = "router-v1-20260501"


# ─────────────────────────────────────────────────────────────
# Verdict extraction
# ─────────────────────────────────────────────────────────────
def extract_verdict(text: str) -> str:
    if not text:
        return "Incorrect"
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
DAY_KEYWORDS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def parse_state(text: str) -> dict[str, set[tuple]]:
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
    a = parse_state(before_text)
    b = parse_state(after_text)
    days = sorted(set(a) | set(b))
    out = []
    for d in days:
        ae = a.get(d, set())
        be = b.get(d, set())
        added = sorted(be - ae)
        removed = sorted(ae - be)
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


def filter_state_to_days(text: str, days: list[str]) -> str:
    if not days:
        return text
    keep = set(days)
    out = []
    cur_day = None
    cur_lines: list[str] = []
    for line in text.splitlines():
        m = DAY_RE.match(line)
        if m:
            if cur_day in keep:
                out.append(f"{cur_day}:")
                out.extend(cur_lines)
            cur_day = m.group(1)
            cur_lines = []
        elif cur_day is not None:
            cur_lines.append(line)
    if cur_day in keep:
        out.append(f"{cur_day}:")
        out.extend(cur_lines)
    return "\n".join(out) if out else "(no events on the addressed days)"


def days_in_text(text: str) -> list[str]:
    found = []
    for d in DAY_KEYWORDS:
        if d.lower() in (text or "").lower():
            found.append(d)
    if any(w in (text or "").lower() for w in ["this week", "next week", "tomorrow", "yesterday", "today", "weekend"]):
        return DAY_KEYWORDS
    return found if found else DAY_KEYWORDS


# ─────────────────────────────────────────────────────────────
# CHECKLIST_V2 system prompt (base for all fewshot variants)
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Few-shot example libraries (V1 → V2 → V3 → V4 cumulative)
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Variant builders. `rec` schema:
#   {cat, query, final, expected, before, after}
# Each builder returns (system_prompt, user_prompt, opts).
# ─────────────────────────────────────────────────────────────
def build_fewshot(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec.get("expected") or ""
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


def build_fewshot_v3(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec.get("expected") or ""
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


def build_fewshot_v4_dayfocus(rec: dict) -> tuple[str, str, dict]:
    q = rec["query"]; final = rec["final"]; exp = rec.get("expected") or ""
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


# ─────────────────────────────────────────────────────────────
# Router: per-category dispatch (per-category bests from the
# 2026-04-30 prompt-tuning sweep — see docs/judge/prompt_tuning.md).
# ─────────────────────────────────────────────────────────────
ROUTER_MAP = {
    "Complex Logic & Conflict (Advanced)":              build_fewshot,            # 92.86
    "Human Chaos (Edge Cases/Fragments)":               build_fewshot_v3,         # 95.65
    "Information Retrieval (Querying)":                 build_fewshot_v3,         # 97.62
    "Modifier & Correction (Rescheduling/Updates)":     build_fewshot_v3,         # 97.56
    "Relative Time References (today, tomorrow, yesterday, this week)": build_fewshot,           # 94.74
    "Schedule a Single Event":                          build_fewshot_v3,         # 86.84
    "Vague & Contextual (Reasoning Required)":          build_fewshot_v4_dayfocus,  # 97.37
}


def build_router(rec: dict) -> tuple[str, str, dict]:
    """Per-category dispatch to the variant that scored best on that category."""
    builder = ROUTER_MAP.get(rec["cat"], build_fewshot_v3)
    return builder(rec)
