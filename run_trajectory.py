#!/usr/bin/env python3
"""Run an agent trajectory against a loaded calendar.

Loads a calendar from data/json_calender/<index>.txt and the corresponding
queries from data/queries/<index>.txt. For each query, runs an agentic loop
where the model uses calendar tools to fulfill the request.

Usage:
    python run_trajectory.py 0
    python run_trajectory.py 0 --query-index 3
    python run_trajectory.py 0 --model gemini-2.0-flash-001 --max-turns 15
"""

import argparse
import json
import os
import sys
import uuid
import warnings

warnings.filterwarnings("ignore")

import vertexai
from vertexai.generative_models import FunctionDeclaration, GenerativeModel, Part, Tool

from environment.environment import CalendarEnvironment

# ── Paths ────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
JSON_CALENDAR_DIR = os.path.join(DATA_DIR, "json_calender")
QUERY_DIR = os.path.join(DATA_DIR, "queries")


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
            return env.get_current_time()

        elif name == "list_events":
            return env.list_events(
                time_min=args.get("time_min"),
                time_max=args.get("time_max"),
            )

        elif name == "get_event":
            return env.get_event(args["event_id"])

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
            return env.create_event(
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
            return env.update_event(
                event_id=args["event_id"],
                updates=updates,
            )

        elif name == "delete_event":
            return env.delete_event(args["event_id"])

        elif name == "respond_to_event":
            result = env.respond_to_event(
                event_id=args["event_id"],
                attending=args["attending"],
            )
            return (
                result if result else {"status": "ok", "attending": args["attending"]}
            )

        else:
            return {
                "error": {"type": "UnknownTool", "message": f"Unknown tool: {name}"}
            }

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
    """Group snapshot events by day name, filtered to only the given days.

    Returns {day_name: [sorted list of event dicts]} for display and data.
    """
    days_set = set(days)
    by_day = {d: [] for d in DAY_NAMES if d in days_set}
    for eid, e in snap.items():
        if e["day"] in days_set:
            by_day[e["day"]].append({**e, "id": eid})
    # Sort events within each day by start time
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


# ── Core Agent Loop ──────────────────────────────────────────


def run_query(
    model, env: CalendarEnvironment, query: str, max_turns: int
) -> list[dict]:
    """Run a single query through the agentic tool-use loop.

    Returns:
        trajectory: list of dicts recording every step.
    """
    trajectory = []
    chat = model.start_chat()

    print(f"\n  {C.BLUE}[USER]       {query}{C.RESET}")
    trajectory.append({"role": "user", "content": query})

    try:
        response = chat.send_message(query)
    except Exception as e:
        print(f"  [ERROR]      Model call failed: {e}")
        trajectory.append({"role": "error", "content": str(e)})
        return trajectory

    for turn in range(1, max_turns + 1):
        print(f"\n  -- Turn {turn} --")

        # Parse model response into function calls and text.
        # The Vertex AI SDK raises AttributeError when accessing .text on a
        # function_call part (and vice versa), so we must guard each access.
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

        # Show any text the model produced
        if text_parts:
            combined = "\n".join(text_parts)
            print_agent_text(combined, is_final=not function_calls)
            trajectory.append({"role": "assistant", "content": combined})

        # If no tool calls, the agent is done
        if not function_calls:
            break

        # Execute each tool call and collect responses
        response_parts = []
        for fc in function_calls:
            args = dict(fc.args)
            print_tool_call(fc.name, args)

            result = dispatch_tool_call(env, fc.name, args)
            if result is None:
                result = {"status": "ok"}

            # Ensure JSON-serializable (handles datetime objects etc.)
            result = json.loads(json.dumps(result, default=str))

            print_tool_result(result)
            trajectory.append(
                {
                    "role": "tool_call",
                    "name": fc.name,
                    "args": args,
                    "result": result,
                }
            )

            response_parts.append(
                Part.from_function_response(name=fc.name, response=result)
            )

        try:
            response = chat.send_message(response_parts)
        except Exception as e:
            print(f"  [ERROR]      Model call failed: {e}")
            trajectory.append({"role": "error", "content": str(e)})
            break
    else:
        print(f"\n  [MAX TURNS]  Reached limit of {max_turns} turns.")

    return trajectory


# ── Data Loading ─────────────────────────────────────────────


def load_calendar_and_queries(index: int):
    """Load calendar events and queries for the given index."""
    cal_path = os.path.join(JSON_CALENDAR_DIR, f"{index}.txt")
    query_path = os.path.join(QUERY_DIR, f"{index}.txt")

    if not os.path.exists(cal_path):
        sys.exit(f"Calendar file not found: {cal_path}")
    if not os.path.exists(query_path):
        sys.exit(f"Query file not found: {query_path}")

    events = CalendarEnvironment.load_json_calendar(cal_path)

    with open(query_path) as f:
        queries = json.load(f)

    # Derive a fallback 'now' from the earliest event (Monday 08:00)
    from datetime import datetime

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
    """Get the simulated 'now' for a query, using its current_time or the fallback.

    Normalises ISO 8601 'T' separator to a space so the environment can parse it.
    """
    ct = query.get("current_time")
    if ct:
        return ct.replace("T", " ")
    return fallback_now


# ── Evaluation ───────────────────────────────────────────────

EVAL_SYSTEM_PROMPT = """\
You evaluate a calendar assistant that has tools to search, create, update, \
and delete calendar events. Judge whether it completed the user's task.

Rules:
- The assistant MUST use its tools to look up calendar data before responding. \
Asking the user for information that is already on the calendar is Incorrect.
- For action tasks (create/update/delete): the calendar state AFTER must \
reflect the expected changes. No change when one was expected = Incorrect.
- For info tasks (queries/lookups): the response must match the calendar data \
and the expected behavior.
- Partial completion is Incorrect.

On the very last line output exactly one word:
Correct
Incorrect
"""


def format_day_state_text(by_day: dict) -> str:
    """Render a day-grouped snapshot as plain text for the eval prompt."""
    lines = []
    for day in DAY_NAMES:
        if day not in by_day:
            continue
        events = by_day[day]
        lines.append(f"{day}:")
        if not events:
            lines.append("  (no events)")
        for e in events:
            start_t = e["start"].split(" ")[1][:5]
            end_t = e["end"].split(" ")[1][:5]
            att = f"  [{', '.join(e['attendees'])}]" if e["attendees"] else ""
            lines.append(f"  {start_t}-{end_t}  {e['summary']}{att}")
    return "\n".join(lines) if lines else "(no relevant events)"


def evaluate_trajectory(
    eval_model,
    query: str,
    final_output: str,
    expected: str,
    before_days: dict,
    after_days: dict,
) -> str:
    """Ask the model to evaluate whether the trajectory was correct.

    Returns one of: 'Correct', 'Incorrect'.
    """
    before_text = format_day_state_text(before_days)
    after_text = format_day_state_text(after_days)

    prompt = f"""\
Query: {query}

Response: {final_output if final_output else "(no response)"}

Expected: {expected if expected else "(not specified)"}

Before:
{before_text}

After:
{after_text}

Was the task completed correctly? End with one word: Correct or Incorrect."""

    # Print eval prompt in purple
    print(f"  {C.MAGENTA}[EVAL PROMPT]{C.RESET}")
    for line in prompt.split("\n"):
        print(f"  {C.MAGENTA}  {line}{C.RESET}")

    try:
        response = eval_model.generate_content(prompt)
        verdict = response.text.strip()
        print(f"\n\n  {C.MAGENTA}[EVAL RAW]    {verdict}{C.RESET}")
        # Scan from the last line upward for an exact verdict word.
        # Incorrect must be checked before Correct to avoid substring false match.
        lines = [l.strip() for l in verdict.splitlines() if l.strip()]
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if line_lower == token.lower():
                    return token
        # Fallback: substring scan from end (Incorrect before Correct)
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if token.lower() in line_lower:
                    return token
        return "Incorrect"
    except Exception as e:
        print(f"  [EVAL ERROR]  {e}")
        return "Incorrect"


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


# ── Main ─────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run agent trajectories on calendar data."
    )
    parser.add_argument(
        "calendar_index", type=int, help="Index of the calendar/query pair (0-49)."
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("GCP_PROJECT", "internal-ml-exp"),
        help="GCP project ID (default: $GCP_PROJECT or 'internal-ml-exp').",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("GCP_LOCATION", "us-central1"),
        help="GCP location (default: $GCP_LOCATION or 'us-central1').",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash-001",
        help="Gemini model name (default: gemini-2.0-flash-001).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max tool-use turns per query (default: 10).",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=None,
        help="Run only a specific query by index (0-based).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save full trajectories to this JSON file.",
    )
    args = parser.parse_args()

    # Load data
    events, queries, fallback_now = load_calendar_and_queries(args.calendar_index)

    # Init Vertex AI (load credentials from JSON file if available)
    _gcp_credentials = None
    _creds_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gcloud_credentials.json"
    )
    if os.path.exists(_creds_path):
        from google.oauth2.credentials import Credentials as OAuth2Credentials

        with open(_creds_path) as _f:
            _cred_data = json.load(_f)
        _gcp_credentials = OAuth2Credentials(
            token=None,
            refresh_token=_cred_data["refresh_token"],
            client_id=_cred_data["client_id"],
            client_secret=_cred_data["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
        )
    vertexai.init(
        project=args.project, location=args.location, credentials=_gcp_credentials
    )
    model = GenerativeModel(
        args.model,
        tools=[CALENDAR_TOOL],
        system_instruction=[SYSTEM_PROMPT],
    )
    eval_model = GenerativeModel(
        args.model,
        system_instruction=[EVAL_SYSTEM_PROMPT],
    )

    # Header
    print_separator()
    print(f"  Trajectory Runner | Calendar {args.calendar_index} | Model: {args.model}")
    print_separator()
    print(f"  Events loaded : {len(events)}")
    print(f"  Queries       : {len(queries)}")
    print(f"  Fallback now  : {fallback_now}")

    # Select queries to run
    if args.query_index is not None:
        if args.query_index >= len(queries):
            sys.exit(
                f"Query index {args.query_index} out of range (0-{len(queries)-1})."
            )
        selected = [(args.query_index, queries[args.query_index])]
    else:
        selected = list(enumerate(queries))

    all_trajectories = []

    for qi, q in selected:
        # Fresh environment for each query, using per-query current_time
        now = get_query_now(q, fallback_now)
        env = CalendarEnvironment()
        env.initialize(events=events, now=now)

        category = q.get("category", "N/A")
        complexity = q.get("complexity", "N/A")
        query_text = q["query"]
        expected = q.get("expected_behavior", "")

        print()
        print_separator("-")
        print(
            f"  QUERY {qi + 1}/{len(queries)} | {category} | Complexity: {complexity}"
        )
        print(f"  Simulated now : {now}")
        if expected:
            print(f"  Expected: {expected}")
        print_separator("-")

        addressed_days = q.get("addressed_days", [])
        # When addressed_days is empty, show all days so eval has full context
        display_days = addressed_days if addressed_days else DAY_NAMES

        before = snapshot_events(env)
        before_days = filter_by_days(before, display_days)

        # Print BEFORE state
        label = ', '.join(addressed_days) if addressed_days else "all days"
        print(f"\n  [BEFORE] Calendar state for: {label}")
        for line in format_day_state(before_days):
            print(f"  {line}")

        trajectory = run_query(model, env, query_text, args.max_turns)

        after = snapshot_events(env)
        after_days = filter_by_days(after, display_days)

        # Print AFTER state
        print(f"\n  [AFTER] Calendar state for: {label}")
        for line in format_day_state(after_days):
            print(f"  {line}")

        # Print diff summary
        changes = diff_snapshots(before, after)
        if changes:
            print(f"\n  [CHANGES]")
            for line in changes:
                print(f"  {line}")
        else:
            print(f"\n  [CHANGES]  (none)")

        # Evaluation
        final_output = next(
            (
                step["content"]
                for step in reversed(trajectory)
                if step["role"] == "assistant"
            ),
            "",
        )
        print()
        print_separator("·")
        verdict = evaluate_trajectory(
            eval_model,
            query_text,
            final_output,
            expected,
            before_days,
            after_days,
        )
        verdict_color = (
            C.GREEN
            if verdict == "Correct"
            else C.RED if verdict == "Incorrect" else C.YELLOW
        )
        print(f"\n  {verdict_color}[EVAL RESULT] {verdict}{C.RESET}")

        all_trajectories.append(
            {
                "query_index": qi,
                "category": category,
                "complexity": complexity,
                "query": query_text,
                "simulated_now": now,
                "expected_behavior": expected,
                "addressed_days": addressed_days,
                "calendar_before": before_days,
                "calendar_after": after_days,
                "state_changes": changes,
                "trajectory": trajectory,
                "eval_verdict": verdict,
            }
        )

    # Summary
    print()
    print_separator()
    print(f"  Done. Ran {len(selected)} queries.")
    verdicts = [t["eval_verdict"] for t in all_trajectories]
    correct = verdicts.count("Correct")
    incorrect = verdicts.count("Incorrect")
    unsure = verdicts.count("Unsure")
    print(
        f"  Eval results  : {C.GREEN}{correct} Correct{C.RESET}  "
        f"{C.RED}{incorrect} Incorrect{C.RESET}  "
        f"{C.YELLOW}{unsure} Unsure{C.RESET}"
    )
    print_separator()

    # Save if requested
    if args.save:
        with open(args.save, "w") as f:
            json.dump(all_trajectories, f, indent=2, default=str)
        print(f"  Trajectories saved to {args.save}")


if __name__ == "__main__":
    main()
