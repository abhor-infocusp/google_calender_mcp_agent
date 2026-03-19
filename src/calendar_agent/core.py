"""Core agent constants, tool declarations, dispatch, and snapshot utilities."""

import json
import sys
import uuid
from datetime import datetime

from vertexai.generative_models import FunctionDeclaration, Tool

from calendar_agent.environment.environment import CalendarEnvironment
from calendar_agent.paths import DATA_DIR


# ── Paths ────────────────────────────────────────────────────

JSON_CALENDAR_DIR = DATA_DIR / "json_calender"
QUERY_DIR = DATA_DIR / "queries"


# ── ANSI Colors ──────────────────────────────────────────────


class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"


# ── Tool Declarations ───────────────────────────────────────

TOOL_DECLARATIONS = [
    FunctionDeclaration(
        name="get_current_time",
        description="Get the current simulated date, time, and day of the week.",
        parameters={"type": "object", "properties": {}},
    ),
    FunctionDeclaration(
        name="list_events",
        description=(
            "List calendar events, optionally filtered by a time range. "
            "Returns all events if no filters are provided."
        ),
        parameters={
            "type": "object",
            "properties": {
                "time_min": {
                    "type": "string",
                    "description": "Start of time range filter, in 'YYYY-MM-DD HH:MM:SS' format.",
                },
                "time_max": {
                    "type": "string",
                    "description": "End of time range filter, in 'YYYY-MM-DD HH:MM:SS' format.",
                },
            },
        },
    ),
    FunctionDeclaration(
        name="get_event",
        description="Get full details of a specific calendar event by its ID.",
        parameters={
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The unique identifier of the event.",
                },
            },
            "required": ["event_id"],
        },
    ),
    FunctionDeclaration(
        name="create_event",
        description="Create a new calendar event.",
        parameters={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Title of the event.",
                },
                "start": {
                    "type": "string",
                    "description": "Start time in 'YYYY-MM-DD HH:MM:SS' format.",
                },
                "end": {
                    "type": "string",
                    "description": "End time in 'YYYY-MM-DD HH:MM:SS' format.",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description or notes.",
                },
                "attendees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of attendee email addresses.",
                },
            },
            "required": ["summary", "start", "end"],
        },
    ),
    FunctionDeclaration(
        name="update_event",
        description="Update fields of an existing calendar event. Only provide the fields you want to change.",
        parameters={
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The unique identifier of the event to update.",
                },
                "summary": {
                    "type": "string",
                    "description": "New title for the event.",
                },
                "start": {
                    "type": "string",
                    "description": "New start time in 'YYYY-MM-DD HH:MM:SS' format.",
                },
                "end": {
                    "type": "string",
                    "description": "New end time in 'YYYY-MM-DD HH:MM:SS' format.",
                },
                "description": {
                    "type": "string",
                    "description": "New description for the event.",
                },
            },
            "required": ["event_id"],
        },
    ),
    FunctionDeclaration(
        name="delete_event",
        description="Delete a calendar event by its ID.",
        parameters={
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The unique identifier of the event to delete.",
                },
            },
            "required": ["event_id"],
        },
    ),
    FunctionDeclaration(
        name="respond_to_event",
        description="Respond to a calendar event invitation.",
        parameters={
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": "The unique identifier of the event.",
                },
                "attending": {
                    "type": "string",
                    "description": "Your RSVP. Must be one of: ACCEPT, DECLINE, MAYBE, NO RESPONSE.",
                    "enum": ["ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE"],
                },
            },
            "required": ["event_id", "attending"],
        },
    ),
]

CALENDAR_TOOL = Tool(function_declarations=TOOL_DECLARATIONS)


# ── Tool Dispatch ────────────────────────────────────────────


def dispatch_tool_call(env: CalendarEnvironment, name: str, args: dict) -> dict:
    """Execute a tool call against the calendar environment and return the result."""
    try:
        if name == "get_current_time":
            result = env.get_current_time()

        elif name == "list_events":
            result = env.list_events(
                time_min=args.get("time_min"),
                time_max=args.get("time_max"),
            )

        elif name == "get_event":
            result = env.get_event(args["event_id"])

        elif name == "create_event":
            emails = args.get("attendees", [])
            attendee_jsons = [
                json.dumps(
                    {
                        "user": {
                            "id": f"user_{uuid.uuid4().hex[:8]}",
                            "name": email.split("@")[0],
                            "email": email,
                        },
                        "attending": "ACCEPT",
                    }
                )
                for email in emails
            ]
            result = env.create_event(
                summary=args["summary"],
                start=args["start"],
                end=args["end"],
                attendees=attendee_jsons,
                description=args.get("description", ""),
            )

        elif name == "update_event":
            updates = {}
            for field in ("summary", "start", "end", "description"):
                if field in args:
                    updates[field] = args[field]
            result = env.update_event(
                event_id=args["event_id"],
                updates=updates,
            )

        elif name == "delete_event":
            result = env.delete_event(args["event_id"])

        elif name == "respond_to_event":
            result = env.respond_to_event(
                event_id=args["event_id"],
                attending=args["attending"],
            )
            if not result:
                result = {"status": "ok", "attending": args["attending"]}

        else:
            result = {
                "error": {"type": "UnknownTool", "message": f"Unknown tool: {name}"}
            }

        return result

    except Exception as e:
        return {"error": {"type": type(e).__name__, "message": str(e)}}


# ── Display Helpers ──────────────────────────────────────────


def fmt_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f'{k}="{v}"')
        else:
            parts.append(f"{k}={json.dumps(v)}")
    return ", ".join(parts)


DAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def snapshot_events(env: CalendarEnvironment) -> dict[str, dict]:
    """Capture current calendar state as {event_id: {day, summary, start, end, attending, attendees}}."""
    snap = {}
    for e in env.calendar.events:
        snap[e.id] = {
            "day": DAY_NAMES[e.start.weekday()],
            "summary": e.summary,
            "start": e.start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": e.end.strftime("%Y-%m-%d %H:%M:%S"),
            "attending": e.attending,
            "attendees": [a.user.email for a in e.attendees],
        }
    return snap


def filter_by_days(snap: dict, days: list[str]) -> dict[str, list[dict]]:
    """Group snapshot events by day name, filtered to only the given days."""
    days_set = set(days)
    by_day = {d: [] for d in DAY_NAMES if d in days_set}
    for eid, e in snap.items():
        if e["day"] in days_set:
            by_day[e["day"]].append({**e, "id": eid})
    for d in by_day:
        by_day[d].sort(key=lambda ev: ev["start"])
    return by_day


def format_day_state(by_day: dict[str, list[dict]]) -> list[str]:
    """Render a day-grouped snapshot into printable lines."""
    lines = []
    for day in DAY_NAMES:
        if day not in by_day:
            continue
        events = by_day[day]
        lines.append(f"    {day}:")
        if not events:
            lines.append("      (no events)")
        for e in events:
            start_t = e["start"].split(" ")[1][:5]
            end_t = e["end"].split(" ")[1][:5]
            att = f"  [{', '.join(e['attendees'])}]" if e["attendees"] else ""
            lines.append(f"      {start_t}-{end_t}  {e['summary']}{att}")
    return lines


def diff_snapshots(before: dict, after: dict) -> list[str]:
    """Compare two event snapshots and return human-readable change lines."""
    changes = []
    before_ids = set(before.keys())
    after_ids = set(after.keys())

    for eid in sorted(after_ids - before_ids):
        e = after[eid]
        changes.append(f"  + CREATED  '{e['summary']}' ({e['start']} -> {e['end']})")

    for eid in sorted(before_ids - after_ids):
        e = before[eid]
        changes.append(f"  - DELETED  '{e['summary']}' ({e['start']} -> {e['end']})")

    for eid in sorted(before_ids & after_ids):
        b, a = before[eid], after[eid]
        for field in ("summary", "start", "end", "attending", "attendees"):
            if b[field] != a[field]:
                changes.append(
                    f"  ~ UPDATED  '{b['summary']}' : {field}: {b[field]} -> {a[field]}"
                )

    return changes


def print_separator(char="=", width=72):
    print(char * width)


def print_tool_call(name: str, args: dict):
    print(f"  {C.YELLOW}[TOOL CALL]  {name}({fmt_args(args)}){C.RESET}")


def print_tool_result(result: dict):
    result_str = json.dumps(result, indent=2, default=str)
    lines = result_str.split("\n")
    if len(lines) > 20:
        lines = lines[:18] + [f"  ... ({len(lines) - 18} more lines)"]
    for i, line in enumerate(lines):
        prefix = "[RESULT]    " if i == 0 else "            "
        print(f"  {prefix}{line}")


def print_agent_text(text: str, is_final: bool):
    label = "[RESPONSE] " if is_final else "[THINKING] "
    color = C.GREEN if is_final else ""
    reset = C.RESET if color else ""
    for i, line in enumerate(text.strip().split("\n")):
        prefix = label if i == 0 else "            "
        print(f"  {color}{prefix}{line}{reset}")


def print_prompt_sent(label: str, content: str):
    """Print what was sent to the model, in blue."""
    print(f"  {C.BLUE}[PROMPT -> MODEL] {label}{C.RESET}")
    for line in content.strip().split("\n"):
        print(f"  {C.BLUE}  {line}{C.RESET}")


# ── Data Loading ─────────────────────────────────────────────


def load_calendar_and_queries(index: int, data_dir=None):
    """Load calendar events and queries for the given index."""
    if data_dir is None:
        data_dir = DATA_DIR
    else:
        from pathlib import Path
        data_dir = Path(data_dir)

    json_cal_dir = data_dir / "json_calender"
    query_dir = data_dir / "queries"
    cal_path = json_cal_dir / f"{index}.txt"
    query_path = query_dir / f"{index}.txt"

    if not cal_path.exists():
        sys.exit(f"Calendar file not found: {cal_path}")
    if not query_path.exists():
        sys.exit(f"Query file not found: {query_path}")

    events = CalendarEnvironment.load_json_calendar(str(cal_path))

    with open(query_path) as f:
        queries = json.load(f)

    earliest = None
    for evt in events:
        dt = datetime.fromisoformat(evt["start"])
        if earliest is None or dt < earliest:
            earliest = dt
    fallback_now = earliest.replace(hour=8, minute=0, second=0).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    return events, queries, fallback_now


def get_query_now(query: dict, fallback_now: str) -> str:
    """Get the simulated 'now' for a query, using its current_time or the fallback."""
    ct = query.get("current_time")
    if ct:
        return ct.replace("T", " ")
    return fallback_now


# ── System Prompt ────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a calendar assistant. You manage the user's Google Calendar using the provided tools.

Rules:
- Always call get_current_time first, then always call list_events to check the calendar before responding.
- Use tool results to get dates, event IDs, and details instead of asking the user.
- Act directly: create, update, or delete events based on the request. Don't ask for confirmation or optional details (e.g. attendee emails).
- Only ask for clarification when truly ambiguous (e.g. multiple events match and it's unclear which one).
- All datetime arguments must be in 'YYYY-MM-DD HH:MM:SS' format.
- After completing an action, confirm what you did.
"""
