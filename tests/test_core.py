import json

import pytest

from calendar_agent.core import (
    CALENDAR_TOOL,
    DAY_NAMES,
    SYSTEM_PROMPT,
    TOOL_DECLARATIONS,
    compute_fallback_now,
    diff_snapshots,
    dispatch_tool_call,
    filter_by_days,
    format_day_state,
    format_tool_result,
    snapshot_events,
)


# ── format_tool_result ─────────────────────────────────────

def test_format_tool_result_none():
    assert format_tool_result(None) == "ok"


def test_format_tool_result_str_passthrough():
    assert format_tool_result("hello") == "hello"
    assert format_tool_result("") == ""


def test_format_tool_result_dict_is_json():
    out = format_tool_result({"a": 1, "b": [2, 3]})
    assert json.loads(out) == {"a": 1, "b": [2, 3]}


def test_format_tool_result_list_is_json():
    out = format_tool_result([1, "x", {"k": "v"}])
    assert json.loads(out) == [1, "x", {"k": "v"}]


# ── DAY_NAMES ──────────────────────────────────────────────

def test_day_names():
    assert DAY_NAMES == [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]


# ── TOOL_DECLARATIONS ──────────────────────────────────────

EXPECTED_TOOL_NAMES = {
    "get_current_time", "list_events", "get_event",
    "create_event", "update_event", "delete_event", "respond_to_event",
}


def test_tool_declarations_count_and_names():
    assert len(TOOL_DECLARATIONS) == 7
    names = {fd.to_dict()["name"] for fd in TOOL_DECLARATIONS}
    assert names == EXPECTED_TOOL_NAMES


def test_calendar_tool_wraps_declarations():
    # Tool from vertexai wraps the declarations; make sure it was constructed
    assert CALENDAR_TOOL is not None


# ── SYSTEM_PROMPT ──────────────────────────────────────────

def test_system_prompt_mentions_get_current_time():
    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT.strip()
    assert "get_current_time" in SYSTEM_PROMPT


# ── snapshot_events ────────────────────────────────────────

def test_snapshot_events_shape(populated_env):
    snap = snapshot_events(populated_env)
    assert set(snap.keys()) == {"evt_mon", "evt_tue", "evt_wed_early"}
    mon = snap["evt_mon"]
    assert mon["day"] == "Monday"
    assert mon["summary"] == "Monday Sync"
    assert mon["start"] == "2026-01-05 10:00:00"
    assert mon["end"] == "2026-01-05 11:00:00"
    assert mon["attending"] == "ACCEPT"
    assert mon["attendees"] == ["alice@example.com"]
    assert snap["evt_tue"]["attendees"] == []


# ── filter_by_days ─────────────────────────────────────────

def test_filter_by_days_groups_and_sorts(populated_env):
    snap = snapshot_events(populated_env)
    # Add a second Wednesday event to verify sorting
    snap["evt_wed_late"] = {
        "day": "Wednesday",
        "summary": "Wed Late",
        "start": "2026-01-07 16:00:00",
        "end": "2026-01-07 17:00:00",
        "attending": "ACCEPT",
        "attendees": [],
    }
    by_day = filter_by_days(snap, ["Monday", "Wednesday"])
    assert set(by_day.keys()) == {"Monday", "Wednesday"}
    assert [e["id"] for e in by_day["Wednesday"]] == ["evt_wed_early", "evt_wed_late"]
    assert [e["id"] for e in by_day["Monday"]] == ["evt_mon"]


def test_filter_by_days_empty_for_unrequested_day(populated_env):
    snap = snapshot_events(populated_env)
    by_day = filter_by_days(snap, ["Friday"])
    assert by_day == {"Friday": []}


# ── format_day_state ───────────────────────────────────────

def test_format_day_state_includes_day_and_event(populated_env):
    snap = snapshot_events(populated_env)
    by_day = filter_by_days(snap, ["Monday", "Tuesday"])
    lines = format_day_state(by_day)
    assert any("Monday" in l for l in lines)
    assert any("Monday Sync" in l for l in lines)
    assert any("Tuesday" in l for l in lines)


def test_format_day_state_empty_input():
    assert format_day_state({}) == []


def test_format_day_state_no_events_marker():
    lines = format_day_state({"Friday": []})
    assert any("(no events)" in l for l in lines)


# ── diff_snapshots ─────────────────────────────────────────

def _evt(summary="X", start="2026-01-05 10:00:00", end="2026-01-05 11:00:00",
        attending="ACCEPT", attendees=None):
    return {
        "day": "Monday",
        "summary": summary,
        "start": start,
        "end": end,
        "attending": attending,
        "attendees": attendees or [],
    }


def test_diff_created():
    changes = diff_snapshots({}, {"e1": _evt(summary="New")})
    assert len(changes) == 1
    assert "CREATED" in changes[0] and "New" in changes[0]


def test_diff_deleted():
    changes = diff_snapshots({"e1": _evt(summary="Old")}, {})
    assert len(changes) == 1
    assert "DELETED" in changes[0] and "Old" in changes[0]


def test_diff_updated():
    before = {"e1": _evt(summary="Same", start="2026-01-05 10:00:00")}
    after = {"e1": _evt(summary="Same", start="2026-01-05 11:00:00")}
    changes = diff_snapshots(before, after)
    assert len(changes) == 1
    assert "UPDATED" in changes[0] and "start" in changes[0]


def test_diff_unchanged_omitted():
    e = _evt()
    assert diff_snapshots({"e1": e}, {"e1": dict(e)}) == []


# ── dispatch_tool_call ─────────────────────────────────────

def test_dispatch_get_current_time(populated_env):
    out = dispatch_tool_call(populated_env, "get_current_time", {})
    assert "Monday" in out
    assert "2026-01-05" in out


def test_dispatch_list_events(populated_env):
    out = dispatch_tool_call(populated_env, "list_events", {})
    assert "Found 3 events" in out


def test_dispatch_get_event(populated_env):
    out = dispatch_tool_call(populated_env, "get_event", {"event_id": "evt_mon"})
    assert "Monday Sync" in out
    assert "evt_mon" in out


def test_dispatch_create_event_wraps_attendee_emails(populated_env):
    out = dispatch_tool_call(populated_env, "create_event", {
        "summary": "New Meet",
        "start": "2026-01-08 10:00:00",
        "end": "2026-01-08 11:00:00",
        "attendees": ["bob@example.com"],
    })
    assert "Event created successfully" in out
    new_event = next(e for e in populated_env.calendar.events if e.summary == "New Meet")
    assert len(new_event.attendees) == 1
    assert new_event.attendees[0].user.email == "bob@example.com"
    assert new_event.attendees[0].user.name == "bob"


def test_dispatch_update_event(populated_env):
    out = dispatch_tool_call(populated_env, "update_event", {
        "event_id": "evt_mon",
        "summary": "Renamed",
    })
    assert "Event updated successfully" in out
    e = next(x for x in populated_env.calendar.events if x.id == "evt_mon")
    assert e.summary == "Renamed"


def test_dispatch_delete_event(populated_env):
    out = dispatch_tool_call(populated_env, "delete_event", {"event_id": "evt_tue"})
    assert "Event deleted" in out
    assert all(e.id != "evt_tue" for e in populated_env.calendar.events)


def test_dispatch_respond_to_event(populated_env):
    out = dispatch_tool_call(populated_env, "respond_to_event", {
        "event_id": "evt_mon",
        "attending": "DECLINE",
    })
    assert "RSVP updated to DECLINE" in out
    e = next(x for x in populated_env.calendar.events if x.id == "evt_mon")
    assert e.attending == "DECLINE"


def test_dispatch_unknown_tool(populated_env):
    out = dispatch_tool_call(populated_env, "no_such_tool", {})
    assert "Unknown tool" in out


def test_dispatch_catches_exception(populated_env):
    # missing required arg should be caught and returned as Error string
    out = dispatch_tool_call(populated_env, "get_event", {})
    assert out.startswith("Error:")


# ── compute_fallback_now ───────────────────────────────────

def test_compute_fallback_now(tmp_path):
    cal = {
        "Monday": [
            {
                "summary": "First",
                "start": "2026-03-09T10:30:00",
                "end": "2026-03-09T11:30:00",
                "attendees": [],
            }
        ],
        "Tuesday": [
            {
                "summary": "Second",
                "start": "2026-03-10T08:00:00",
                "end": "2026-03-10T09:00:00",
                "attendees": [],
            }
        ],
    }
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    out = compute_fallback_now(str(p))
    assert out == "2026-03-09 08:00:00"
