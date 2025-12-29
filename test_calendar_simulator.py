import pytest
from calender import CalendarEnvironment

def test_list_calendars():
    env = CalendarEnvironment()
    result = env.list_calendars()

    assert "calendars" in result
    assert len(result["calendars"]) == 1
    assert result["calendars"][0]["id"] == "primary"
    assert result["calendars"][0]["primary"] is True


def test_create_event():
    env = CalendarEnvironment()

    result = env.create_event(
        calendar_id="primary",
        summary="Team Sync",
        start="2025-12-26T10:00:00Z",
        end="2025-12-26T11:00:00Z",
        attendees=["agent@local"]
    )

    event = result["event"]
    assert event["summary"] == "Team Sync"
    assert event["start"] == "2025-12-26T10:00:00Z"
    assert "evt_" in event["id"]
    assert event["attendees"]["agent@local"] == "needsAction"


def test_list_events():
    env = CalendarEnvironment()

    env.create_event(
        calendar_id="primary",
        summary="Meeting",
        start="2025-12-26T09:00:00Z",
        end="2025-12-26T10:00:00Z"
    )

    result = env.list_events("primary")

    assert len(result["events"]) == 1
    assert result["events"][0]["summary"] == "Meeting"


def test_get_event():
    env = CalendarEnvironment()

    create_result = env.create_event(
        calendar_id="primary",
        summary="1:1",
        start="2025-12-26T12:00:00Z",
        end="2025-12-26T12:30:00Z"
    )

    event_id = create_result["event"]["id"]

    result = env.get_event("primary", event_id)

    assert result["event"]["id"] == event_id
    assert result["event"]["summary"] == "1:1"


def test_update_event():
    env = CalendarEnvironment()

    event_id = env.create_event(
        calendar_id="primary",
        summary="Draft Review",
        start="2025-12-26T13:00:00Z",
        end="2025-12-26T14:00:00Z"
    )["event"]["id"]

    result = env.update_event(
        calendar_id="primary",
        event_id=event_id,
        summary="Final Review",
        location="Zoom"
    )

    event = result["event"]
    assert event["summary"] == "Final Review"
    assert event["location"] == "Zoom"


def test_delete_event():
    env = CalendarEnvironment()

    event_id = env.create_event(
        calendar_id="primary",
        summary="Temp Event",
        start="2025-12-26T15:00:00Z",
        end="2025-12-26T16:00:00Z"
    )["event"]["id"]

    delete_result = env.delete_event("primary", event_id)

    assert delete_result["status"] == "deleted"

    with pytest.raises(ValueError):
        env.get_event("primary", event_id)


def test_respond_to_event():
    env = CalendarEnvironment()

    event_id = env.create_event(
        calendar_id="primary",
        summary="All Hands",
        start="2025-12-26T17:00:00Z",
        end="2025-12-26T18:00:00Z",
        attendees=["agent@local"]
    )["event"]["id"]

    result = env.respond_to_event(
        calendar_id="primary",
        event_id=event_id,
        email="agent@local",
        response_status="accepted"
    )

    assert result["responseStatus"] == "accepted"

    event = env.get_event("primary", event_id)["event"]
    assert event["attendees"]["agent@local"] == "accepted"


def test_get_freebusy():
    env = CalendarEnvironment()

    env.create_event(
        calendar_id="primary",
        summary="Busy Slot",
        start="2025-12-26T10:00:00Z",
        end="2025-12-26T11:00:00Z"
    )

    result = env.get_freebusy(
        calendar_ids=["primary"],
        time_min="2025-12-26T09:00:00Z",
        time_max="2025-12-26T12:00:00Z"
    )

    assert len(result["busy"]) == 1
    assert result["busy"][0]["start"] == "2025-12-26T10:00:00Z"


def test_get_current_time():
    env = CalendarEnvironment()
    result = env.get_current_time()

    assert "now" in result
    assert result["timezone"] == "UTC"


def test_calendar_not_found():
    env = CalendarEnvironment()

    with pytest.raises(ValueError):
        env.list_events("does_not_exist")


def test_event_not_found():
    env = CalendarEnvironment()

    with pytest.raises(ValueError):
        env.get_event("primary", "evt_missing")


