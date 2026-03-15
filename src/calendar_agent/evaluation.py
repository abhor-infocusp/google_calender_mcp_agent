"""Evaluation logic for calendar agent trajectories."""

from calendar_agent.core import C, DAY_NAMES


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

    try:
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("Gemini eval timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(30)
        try:
            response = eval_model.generate_content(prompt)
            verdict = response.text.strip()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        lines = [l.strip() for l in verdict.splitlines() if l.strip()]
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if line_lower == token.lower():
                    return token
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if token.lower() in line_lower:
                    return token
        return "Incorrect"
    except Exception as e:
        print(f"  [EVAL ERROR]  {e}")
        return "Incorrect"
