"""Per-category structured prompts (v3 design).

Each builder takes a record and emits (system_prompt, user_prompt, opts).
The user prompt drops raw BEFORE/AFTER text dumps in favour of pre-computed
structured fields tailored to that category's decisive question.

Fallback: categories not implemented here delegate to the production
build_router (v1) so we get a clean side-by-side per-cat comparison.
"""
from __future__ import annotations

from .features import extract_features
from .prompts import build_router, build_fewshot_v3


# ─────────────────────────────────────────────────────────────────────
# System prompts (per cat). Short, decisive-question framed.
# ─────────────────────────────────────────────────────────────────────
_SYS_BASE = """\
You evaluate a calendar assistant. Decide whether the agent completed the
user's task. The user prompt gives you pre-computed facts:

  - QUERY: the user's exact words. Trust QUERY over EXPECTED when they
    differ — EXPECTED is one approximate interpretation, not gospel.
  - EXPECTED: a reference interpretation of the query.
  - RESPONSE: what the agent said back.
  - AGENT_ACTION: classified action type.
  - CALENDAR_DIFF: atomic event-level changes (or "no calendar change").
    Sections — MOVED (cross-day relocation, treat as a single move),
    MODIFIED (in-place edit), ADDED, REMOVED.
  - RESPONSE_CITATIONS: events the response references, resolved against
    the calendar (BOTH | BEFORE | AFTER | MISSING).
  - RELEVANT_CALENDAR (if present): events on the user's addressed days.

HARD-FAIL RULE (overrides all category checks):
  If RESPONSE_WELL_FORMED is False (e.g. raw <tool_call> XML leaked,
  garbled non-English mojibake, or empty response), the verdict is
  Incorrect — the agent failed to produce a usable final answer.

Reason carefully but concisely. End your response with exactly one word
on the last line: Correct or Incorrect.
"""


_SYS_MODIFIER = _SYS_BASE + """
This is a MODIFIER query: the user wants to change an existing event.
Decisive checks:
  1. Did CALENDAR_DIFF.modified contain the event the user referenced?
  2. Does the change (time / attendees / title) match what the user asked?
  3. If only ADDED+REMOVED appear (move across days), treat as a modification.
  4. If RESPONSE claims success but DIFF is empty → Incorrect.
"""

_SYS_CHAOS = _SYS_BASE + """
This is a CHAOS query: the user input is fragmented or edge-case.
Decisive checks:
  1. Use RELEVANT_CALENDAR to identify what event the fragment likely
     refers to (e.g. "kale thing" → "Harvesting mature kale", "yoga
     thing" → "Yoga Class", "Oyster beds Monday" → an oyster-related
     event on Monday).
  2. If a plausible matching event exists in RELEVANT_CALENDAR, the
     agent SHOULD have acted on it. Asking "I cannot find anything"
     when the event is in plain sight is Incorrect.
  3. Asking for clarification is only acceptable when the calendar
     contains MULTIPLE plausible matches OR no plausible match at all.
  4. Acting on the right inferred event → Correct, even if the user's
     input was sparse.
"""

_SYS_COMPLEX = _SYS_BASE + """
This is a COMPLEX query with multiple sub-tasks.
Decisive checks:
  1. Decompose the QUERY (NOT EXPECTED) into atomic sub-tasks.
  2. For each sub-task, check the corresponding CALENDAR_DIFF entry or
     RESPONSE statement.
  3. MOVED entries in CALENDAR_DIFF are single move operations — they
     are NOT duplicate creations. "Moved from Mon to Wed" appearing as
     a MOVED entry means the move succeeded.
  4. Verdict is Correct only if ALL sub-tasks the USER asked for were
     done. If the user did not pin down a time/duration, lean Correct
     when the agent picked a reasonable one.
  5. EXPECTED may name extra constraints the user never stated (specific
     time, specific attendee). Do NOT penalise the agent for skipping
     constraints absent from QUERY.
  6. Modifying the WRONG event, or claiming success without any matching
     state change → Incorrect.
"""


_SYS_IR = _SYS_BASE + """
This is an INFORMATION-RETRIEVAL query: the user wants a specific fact.
Decisive checks:
  1. Read EXPECTED_ANSWER_TYPE — the kind of fact the user requested
     (time, attendees, presence, duration, location, count, listing).
  2. The RESPONSE must NARROW to the asked fact. If the user asked
     "what time is X?" and the agent listed three events including X
     without identifying X's time, that is Incorrect — agent did not
     actually answer the question.
  3. If RESPONSE says "no event" or "didn't find" but RELEVANT_CALENDAR
     contains an obvious match → Incorrect (agent missed it).
  4. Calendar must NOT change for IR queries (DIFF should be empty).
  5. The agent identifying the right event AND providing the requested
     fact (time / attendees / etc.) → Correct.
"""


_SYS_RELTIME = _SYS_BASE + """
This is a RELATIVE-TIME query (today/tomorrow/yesterday/this week/etc.).
Decisive checks:
  1. Did the agent resolve the relative time correctly? (Use the
     CALENDAR's actual content as the reference.)
  2. EXPECTED_ANSWER_TYPE tells you what fact the user wanted.
  3. If the agent returned "no events" but RELEVANT_CALENDAR shows
     events on the resolved date → Incorrect (missed events).
  4. If the agent dumped events without filtering to the user's
     constraint (e.g. "with Jake", "for fun") → Incorrect.
  5. Listing-type queries ("what's on my schedule tomorrow") accept
     a complete listing of the resolved date.
"""


def _addressed_days_block(rec: dict, feats: dict) -> str:
    """When DIFF is empty, surface the relevant calendar window so IR/Vague/
    Chaos judges can verify the response. Compresses event list to one line
    per event."""
    if not feats["before_events"]:
        return "(no events on relevant days)"
    return "\n".join(f"  - {e.fmt()}" for e in feats["before_events"])


def _user_block(rec: dict, feats: dict, *, include_relevant: bool,
                include_answer_type: bool = False) -> str:
    parts: list[str] = [
        f"QUERY: {rec['query']}",
        f"EXPECTED: {rec.get('expected') or '(not specified)'}",
        f"STATE_CHANGE_REQUIRED: {feats['state_change_required']}",
    ]
    if include_answer_type:
        parts.append(f"EXPECTED_ANSWER_TYPE: {feats['expected_answer_type']}")
    parts.extend([
        "",
        f"RESPONSE: {rec.get('final') or '(no response)'}",
        f"AGENT_ACTION: {feats['agent_action']}",
        f"RESPONSE_WELL_FORMED: {feats['response_well_formed']}"
        + (f"  ({feats['response_malformed_reason']})" if not feats['response_well_formed'] else ""),
        "",
        "CALENDAR_DIFF:",
        feats["diff_text"],
    ])
    if feats["citations"]:
        parts.extend(["", "RESPONSE_CITATIONS:", feats["citations_text"]])
    if include_relevant:
        parts.extend(["", "RELEVANT_CALENDAR (events on addressed days, before-state):",
                      _addressed_days_block(rec, feats)])
    parts.extend(["", "Was the task completed correctly?",
                  "End with one word: Correct or Incorrect."])
    return "\n".join(parts)


def build_modifier(rec: dict) -> tuple[str, str, dict]:
    feats = extract_features(rec)
    # Modifier: DIFF is the whole story; relevant calendar is noise.
    return _SYS_MODIFIER, _user_block(rec, feats, include_relevant=False), {}


def build_chaos(rec: dict) -> tuple[str, str, dict]:
    feats = extract_features(rec)
    # Chaos may need calendar context to disambiguate fragments.
    needs_window = feats["diff"].is_empty()
    return _SYS_CHAOS, _user_block(rec, feats, include_relevant=needs_window), {}


def build_complex(rec: dict) -> tuple[str, str, dict]:
    feats = extract_features(rec)
    # Complex: include relevant calendar so judge can verify each sub-step.
    return _SYS_COMPLEX, _user_block(rec, feats, include_relevant=True), {}


def build_ir(rec: dict) -> tuple[str, str, dict]:
    feats = extract_features(rec)
    # IR: include relevant calendar so judge can verify the response is
    # consistent with calendar state, and the answer-type signal.
    return _SYS_IR, _user_block(rec, feats, include_relevant=True,
                                include_answer_type=True), {}


def build_reltime(rec: dict) -> tuple[str, str, dict]:
    feats = extract_features(rec)
    return _SYS_RELTIME, _user_block(rec, feats, include_relevant=True,
                                     include_answer_type=True), {}


# ─────────────────────────────────────────────────────────────────────
# Per-cat router (v3 structured): use new prompts for 5 cats, fall back
# to production `build_router` (v1) for the other 2 — clean A/B per-cat.
# ─────────────────────────────────────────────────────────────────────
ROUTER_MAP_STRUCTURED = {
    "Modifier & Correction (Rescheduling/Updates)":     build_modifier,
    "Human Chaos (Edge Cases/Fragments)":               build_chaos,
    "Complex Logic & Conflict (Advanced)":              build_complex,
    "Information Retrieval (Querying)":                 build_ir,
    "Relative Time References (today, tomorrow, yesterday, this week)": build_reltime,
}


def build_router_structured(rec: dict) -> tuple[str, str, dict]:
    builder = ROUTER_MAP_STRUCTURED.get(rec["cat"])
    if builder is None:
        return build_router(rec)
    return builder(rec)
