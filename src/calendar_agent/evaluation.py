"""Evaluation logic for calendar agent trajectories."""

from calendar_agent.core import C, DAY_NAMES


EVAL_SYSTEM_PROMPT = """\
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
            rsvp = f"  (RSVP: {e['attending']})" if e.get("attending", "ACCEPT") != "ACCEPT" else ""
            lines.append(f"  {start_t}-{end_t}  {e['summary']}{att}{rsvp}")
    return "\n".join(lines) if lines else "(no relevant events)"


def evaluate_trajectory(
    eval_model,
    query: str,
    final_output: str,
    expected: str,
    before_days: dict,
    after_days: dict,
) -> tuple[str, str]:
    """Ask the model to evaluate whether the trajectory was correct.

    Returns (verdict, reasoning) where verdict is 'Correct' or 'Incorrect'.
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
        signal.alarm(60)
        try:
            response = eval_model.generate_content(prompt)
            full_response = response.text.strip()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        lines = [l.strip() for l in full_response.splitlines() if l.strip()]
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if line_lower == token.lower():
                    return token, full_response
        for line in reversed(lines):
            line_lower = line.lower()
            for token in ("Incorrect", "Correct"):
                if token.lower() in line_lower:
                    return token, full_response
        return "Incorrect", full_response
    except Exception as e:
        print(f"  [EVAL ERROR]  {e}")
        return "Incorrect", str(e)
