from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, format_day_state_text


def test_eval_system_prompt_mentions_verdict_tokens():
    assert isinstance(EVAL_SYSTEM_PROMPT, str)
    assert "Correct" in EVAL_SYSTEM_PROMPT
    assert "Incorrect" in EVAL_SYSTEM_PROMPT


def test_format_day_state_text_empty():
    assert format_day_state_text({}) == "(no relevant events)"


def test_format_day_state_text_renders_event():
    by_day = {
        "Monday": [
            {
                "summary": "Standup",
                "start": "2026-01-05 09:00:00",
                "end": "2026-01-05 09:30:00",
                "attendees": ["alice@example.com"],
                "attending": "ACCEPT",
            }
        ]
    }
    out = format_day_state_text(by_day)
    assert "Monday:" in out
    assert "Standup" in out
    assert "09:00-09:30" in out
    assert "alice@example.com" in out
    # ACCEPT is the default, RSVP suffix should be hidden
    assert "RSVP" not in out


def test_format_day_state_text_shows_non_accept_rsvp():
    by_day = {
        "Tuesday": [
            {
                "summary": "Optional Meet",
                "start": "2026-01-06 14:00:00",
                "end": "2026-01-06 15:00:00",
                "attendees": [],
                "attending": "DECLINE",
            }
        ]
    }
    out = format_day_state_text(by_day)
    assert "RSVP: DECLINE" in out


def test_format_day_state_text_orders_by_day_names():
    # Insert in reverse order to confirm rendering follows DAY_NAMES, not insertion
    by_day = {
        "Wednesday": [],
        "Monday": [],
    }
    out = format_day_state_text(by_day)
    assert out.index("Monday:") < out.index("Wednesday:")


def test_format_day_state_text_no_events_marker():
    out = format_day_state_text({"Friday": []})
    assert "Friday:" in out
    assert "(no events)" in out
