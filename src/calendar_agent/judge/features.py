"""Pre-computed structured features for judge prompts.

Takes a record with `before` / `after` (formatted day-state text), `query`,
`expected`, `final`, `cat` and emits structured fields that compress raw
state into the inferences a judge actually needs:

  - diff.added / removed / modified  (event-level, on addressed days)
  - response_citations               (events the response mentions, resolved)
  - agent_action_type                (created | modified | deleted | listed |
                                      asked | refused | errored | unknown)
  - state_change_required            (yes | no | either) from `expected`

The before/after text format is the output of `format_day_state_text`:

    Tuesday:
      08:00-09:00  Team Check-in - Review tasks for the day  [a@b.com]
      09:30-11:00  Harvesting mature kale
    Wednesday:
      ...

Parser is line-based: day headers end with ":", events are indented and
match `HH:MM-HH:MM  Title  [optional, comma, separated, attendees]`.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Iterable

_DAY_HEADER = re.compile(r"^([A-Z][a-z]+):\s*$")
_EVENT_LINE = re.compile(
    r"^\s+(\d{2}:\d{2})-(\d{2}:\d{2})\s+(.+?)\s*(?:\[(.+)\])?\s*$"
)


@dataclass
class Event:
    day: str
    start: str
    end: str
    title: str
    attendees: tuple[str, ...] = ()

    def key(self) -> tuple[str, str, str, str]:
        return (self.day, self.start, self.end, self.title.strip().lower())

    def fmt(self) -> str:
        att = f"  [{', '.join(self.attendees)}]" if self.attendees else ""
        return f"{self.day} {self.start}-{self.end}  {self.title}{att}"


@dataclass
class Diff:
    added: list[Event] = field(default_factory=list)
    removed: list[Event] = field(default_factory=list)
    modified: list[tuple[Event, Event, list[str]]] = field(default_factory=list)
    # modified entries: (before_evt, after_evt, list_of_changed_field_names)
    moved: list[tuple[Event, Event]] = field(default_factory=list)
    # moved entries: same title, different (day, start, end) — cross-day or
    # cross-time pair that previously appeared as separate add+remove.

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.modified or self.moved)


def parse_day_state(text: str) -> list[Event]:
    out: list[Event] = []
    if not text:
        return out
    cur_day = ""
    for raw in text.splitlines():
        if not raw.strip():
            continue
        m = _DAY_HEADER.match(raw)
        if m:
            cur_day = m.group(1)
            continue
        m = _EVENT_LINE.match(raw)
        if m and cur_day:
            start, end, title, atts = m.groups()
            attendees: tuple[str, ...] = ()
            if atts:
                attendees = tuple(a.strip() for a in atts.split(",") if a.strip())
            out.append(Event(day=cur_day, start=start, end=end, title=title.strip(),
                             attendees=attendees))
    return out


def _norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip().lower())


def compute_diff(before: list[Event], after: list[Event]) -> Diff:
    """Diff two event lists. Match by exact (day, start, end, title); pair
    up unmatched same-title events as 'modified'."""
    by_key_b = {e.key(): e for e in before}
    by_key_a = {e.key(): e for e in after}
    added_keys = set(by_key_a) - set(by_key_b)
    removed_keys = set(by_key_b) - set(by_key_a)

    diff = Diff()

    # Pair removed↔added by same normalised title → "modified" or "moved"
    rem_by_title: dict[str, list[Event]] = {}
    for k in removed_keys:
        e = by_key_b[k]
        rem_by_title.setdefault(_norm_title(e.title), []).append(e)
    for k in list(added_keys):
        e = by_key_a[k]
        nt = _norm_title(e.title)
        if rem_by_title.get(nt):
            old = rem_by_title[nt].pop(0)
            # Cross-day pair = MOVED (e.g. Mon → Wed). Same-day = MODIFIED.
            if old.day != e.day:
                diff.moved.append((old, e))
            else:
                changed = []
                if old.start != e.start or old.end != e.end:
                    changed.append("time")
                if set(old.attendees) != set(e.attendees):
                    changed.append("attendees")
                if old.title != e.title:
                    changed.append("title")
                diff.modified.append((old, e, changed))
            added_keys.discard(k)
            removed_keys.discard(old.key())
    diff.added = [by_key_a[k] for k in added_keys]
    diff.removed = [by_key_b[k] for k in removed_keys]
    return diff


def fmt_diff(d: Diff) -> str:
    if d.is_empty():
        return "(no calendar change)"
    parts: list[str] = []
    if d.moved:
        parts.append("MOVED (cross-day relocation, treat as a single move):")
        for old, new in d.moved:
            parts.append(f"  → {old.title}")
            parts.append(f"      from: {old.day} {old.start}-{old.end}")
            parts.append(f"      to:   {new.day} {new.start}-{new.end}")
    if d.added:
        parts.append("ADDED:")
        for e in d.added:
            parts.append(f"  + {e.fmt()}")
    if d.removed:
        parts.append("REMOVED:")
        for e in d.removed:
            parts.append(f"  - {e.fmt()}")
    if d.modified:
        parts.append("MODIFIED (in-place edit):")
        for old, new, fields in d.modified:
            parts.append(f"  ~ {old.title}: changed {', '.join(fields) or '(meta)'}")
            parts.append(f"      from: {old.day} {old.start}-{old.end}"
                         + (f" [{', '.join(old.attendees)}]" if old.attendees else ""))
            parts.append(f"      to:   {new.day} {new.start}-{new.end}"
                         + (f" [{', '.join(new.attendees)}]" if new.attendees else ""))
    return "\n".join(parts)


# ── Response citation analysis ─────────────────────────────
_TIME_REF = re.compile(r"\b(\d{1,2}):(\d{2})\s*(AM|PM|am|pm)?\b|\b(\d{1,2})\s*(AM|PM|am|pm)\b")
_DAY_REF = re.compile(r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|tomorrow|yesterday)\b",
                      re.IGNORECASE)
_QUOTED = re.compile(r"['\"]([^'\"\n]{2,80})['\"]")


def response_citations(final: str, before: list[Event], after: list[Event]) -> list[dict]:
    """Find event references in `final` and check membership in BEFORE/AFTER.

    A citation is a quoted phrase or a day+time pair near a verb. Match by
    title substring (case-insensitive) on event titles. Returns a list of
    dicts: {ref, where: 'BEFORE'|'AFTER'|'BOTH'|'MISSING'|'AMBIGUOUS'}.
    """
    if not final:
        return []
    cites: list[dict] = []
    titles_b = {_norm_title(e.title): e for e in before}
    titles_a = {_norm_title(e.title): e for e in after}

    seen: set[str] = set()
    # Quoted phrases first
    for m in _QUOTED.finditer(final):
        ref = m.group(1).strip()
        nref = _norm_title(ref)
        if nref in seen or len(nref) < 3:
            continue
        seen.add(nref)
        in_b = any(nref in t or t in nref for t in titles_b)
        in_a = any(nref in t or t in nref for t in titles_a)
        where = ("BOTH" if (in_b and in_a) else "BEFORE" if in_b
                 else "AFTER" if in_a else "MISSING")
        cites.append({"ref": ref, "where": where})

    return cites


def fmt_citations(cites: list[dict]) -> str:
    if not cites:
        return "(no specific events cited)"
    return "\n".join(f"  - {c['ref']!r} → {c['where']}" for c in cites)


# ── Agent action classifier ─────────────────────────────
_ASK_RE = re.compile(r"\b(could you|can you|would you|please clarify|which|do you mean|let me know|specify|need more|more info)\b",
                     re.IGNORECASE)
_REFUSE_RE = re.compile(r"\b(cannot|can't|unable to|don't have access|not able)\b", re.IGNORECASE)
_ERR_RE = re.compile(r"\b(error|failed|exception|something went wrong)\b", re.IGNORECASE)


def classify_action(diff: Diff, final: str) -> str:
    if not final:
        return "errored"
    if _ERR_RE.search(final):
        return "errored"
    if final.rstrip().endswith("?") or _ASK_RE.search(final):
        # but if calendar still changed, agent acted before asking
        if not diff.is_empty():
            return "acted_then_asked"
        return "asked"
    if _REFUSE_RE.search(final) and diff.is_empty():
        return "refused"
    if diff.modified and not (diff.added or diff.removed):
        return "modified"
    if diff.added and not diff.removed:
        return "created"
    if diff.removed and not diff.added:
        return "deleted"
    if diff.added and diff.removed:
        return "moved/replaced"
    if diff.is_empty():
        return "listed/info"
    return "mixed"


# ── State-change requirement (from expected text) ───────────
_NEEDS_CHANGE_VERBS = re.compile(
    r"\b(create|schedule|add|update|change|move|reschedule|delete|remove|cancel|decline|accept)\b",
    re.IGNORECASE,
)
_INFO_VERBS = re.compile(r"\b(list|find|return|tell|show|when is|what time|who is)\b",
                         re.IGNORECASE)


def state_change_required(expected: str) -> str:
    """yes | no | either"""
    if not expected:
        return "either"
    has_change = bool(_NEEDS_CHANGE_VERBS.search(expected))
    has_info = bool(_INFO_VERBS.search(expected))
    if has_change and not has_info:
        return "yes"
    if has_info and not has_change:
        return "no"
    if has_change and has_info:
        return "either"
    return "either"


# ── Expected-answer-type classifier ─────────────────────
# For information-retrieval queries: what specific thing is the user asking?
# Helps the judge check "did the response actually narrow to the answer"
# instead of accepting "agent listed lots of events".
_ANSWER_TYPE_PATTERNS = [
    ("time",      re.compile(r"\b(what time|when is|when's|at what time|start time|end time)\b", re.I)),
    ("attendees", re.compile(r"\b(who(?:'s| is)?\s+(?:invited|coming|attending)|who am I .* with|who is on|who's on|attendees|invited)\b", re.I)),
    ("presence",  re.compile(r"\b(do I have|am I (?:free|busy|working)|is there (?:a|an|any)|anything (?:scheduled|on))\b", re.I)),
    ("duration",  re.compile(r"\b(how long|duration|how many (?:minutes|hours))\b", re.I)),
    ("location",  re.compile(r"\b(where (?:is|am I)|location|address)\b", re.I)),
    ("count",     re.compile(r"\b(how many|count of)\b", re.I)),
    ("listing",   re.compile(r"\b(what(?:'s| is) on|what (?:do I|did I|am I doing|does .* look like)|list (?:the|all|my)|what events|schedule|anything (?:scheduled|on))\b", re.I)),
]


def expected_answer_type(query: str) -> str:
    """Classify what kind of answer an IR/Vague/RelTime query is asking for."""
    if not query:
        return "unknown"
    for tag, pat in _ANSWER_TYPE_PATTERNS:
        if pat.search(query):
            return tag
    return "unknown"


# ── Response well-formed check ─────────────────────────
# Catches garbled tool-call leakage, refusals dressed as answers, or
# truncated outputs.
_TOOL_CALL_LEAK = re.compile(r"<tool_call>|</tool_call>|<\|tool_call\|>", re.I)
_NON_ASCII_HEAVY = re.compile(r"[^\x00-\x7f]")


def response_well_formed(final: str) -> tuple[bool, str]:
    """Return (is_well_formed, reason). False means the response itself is
    broken — the judge should hard-fail without checking semantic correctness."""
    if not final or len(final.strip()) < 5:
        return False, "empty or near-empty response"
    if _TOOL_CALL_LEAK.search(final):
        return False, "response leaks raw <tool_call> XML — agent failed to format final answer"
    # Heavy non-ASCII (>20% of chars and >5 chars) usually indicates mojibake.
    nonascii = len(_NON_ASCII_HEAVY.findall(final))
    if nonascii > 5 and nonascii / len(final) > 0.2:
        return False, f"response has heavy non-ASCII characters ({nonascii} of {len(final)}) — likely garbled"
    return True, ""


# ── Top-level extractor ─────────────────────────────────
def extract_features(rec: dict) -> dict:
    """Return all derived signals for a judge record."""
    before = parse_day_state(rec.get("before") or "")
    after = parse_day_state(rec.get("after") or "")
    diff = compute_diff(before, after)
    cites = response_citations(rec.get("final") or "", before, after)
    action = classify_action(diff, rec.get("final") or "")
    needs_change = state_change_required(rec.get("expected") or "")
    answer_type = expected_answer_type(rec.get("query") or "")
    well_formed, wf_reason = response_well_formed(rec.get("final") or "")
    return {
        "before_events": before,
        "after_events": after,
        "diff": diff,
        "diff_text": fmt_diff(diff),
        "citations": cites,
        "citations_text": fmt_citations(cites),
        "agent_action": action,
        "state_change_required": needs_change,
        "expected_answer_type": answer_type,
        "response_well_formed": well_formed,
        "response_malformed_reason": wf_reason,
    }
