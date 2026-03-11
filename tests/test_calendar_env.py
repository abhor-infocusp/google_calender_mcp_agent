import pytest

from datetime import datetime

from environment.environment import CalendarEnvironment

@pytest.fixture
def env():
    env = CalendarEnvironment()
    env.calendar.events = []
    env.users = []
    env.now = None
    return env


@pytest.fixture
def sample_event_dict():
    return {
        "id": "evt_1",
        "summary": "Test Event",
        "start": "2026-01-01 10:00:00",
        "end": "2026-01-01 11:00:00",
        "attendees": [],
        "description": "Test description",
        "optional": False,
    }


@pytest.fixture
def sample_user_dict():
    return {
        "id": "2",
        "name": "Jane Doe",
        "email": "jane.doe@example.com",
    }

# -------------------------
# Initialization
# -------------------------

def test_initialize_success(env, sample_event_dict, sample_user_dict):
    env.initialize(
        events=[sample_event_dict],
        now="2026-01-01 09:00:00",
    )

    assert len(env.calendar.events) == 1
    assert env.calendar.events[0].summary == "Test Event"
    assert env.now == datetime(2026, 1, 1, 9, 0, 0)


def test_initialize_invalid_datetime_raises(env, sample_event_dict, sample_user_dict):
    with pytest.raises(Exception):
        env.initialize(
            events=[sample_event_dict],
            now="invalid-datetime",
        )


# -------------------------
# Datetime parsing
# -------------------------

def test_parse_datetime_success(env):
    dt = env._parse_datetime_str("2026-01-01 10:00:00")
    assert dt == datetime(2026, 1, 1, 10, 0, 0)


def test_parse_datetime_invalid_format(env):
    with pytest.raises(Exception):
        env._parse_datetime_str("2026/01/01")


# -------------------------
# List events
# -------------------------

def test_list_events_no_filters(env, sample_event_dict):
    env.initialize(events=[sample_event_dict], now="2026-01-01 09:00:00")

    result = env.list_events()
    assert "events" in result
    assert len(result["events"]) == 1


def test_list_events_with_time_min(env, sample_event_dict):
    env.initialize(events=[sample_event_dict], now="2026-01-01 09:00:00")

    result = env.list_events(time_min="2026-01-01 10:30:00")
    assert len(result["events"]) == 1


def test_list_events_with_time_max(env, sample_event_dict):
    env.initialize(events=[sample_event_dict], now="2026-01-01 09:00:00")

    result = env.list_events(time_max="2026-01-01 09:30:00")
    assert len(result["events"]) == 0


def test_list_events_invalid_time(env):
    result = env.list_events(time_min="bad-input")
    assert "error" in result


# -------------------------
# Get event
# -------------------------

def test_get_event_success(env, sample_event_dict):
    env.initialize(events=[sample_event_dict], now="2026-01-01 09:00:00")

    result = env.get_event("evt_1")
    assert "event" in result


def test_get_event_not_found(env):
    result = env.get_event("missing")
    assert "error" in result


# -------------------------
# Create event
# -------------------------

def test_create_event_success(env):
    result = env.create_event(
        summary="New Event",
        start="2026-01-01 12:00:00",
        end="2026-01-01 13:00:00",
        attendees=[],
    )

    assert result["message"] == "Event created successfully."
    assert len(env.calendar.events) == 1


def test_create_event_invalid_start(env):
    result = env.create_event(
        summary="Bad Event",
        start="invalid",
        end="2026-01-01 13:00:00",
    )
    assert "error" in result


# -------------------------
# Update event
# -------------------------

def test_update_event_summary(env, sample_event_dict):
    env.initialize(events=[sample_event_dict], now="2026-01-01 09:00:00")

    result = env.update_event(
        event_id="evt_1",
        updates={
            'summary': 'Updated summary'
        },
    )

    assert result["message"] == "Event updated successfully."
    assert env.calendar.events[0].summary == "Updated summary"


def test_update_event_invalid_id(env):
    result = env.update_event(
        event_id="missing",
        updates={"summary": "X"},
    )
    assert "error" in result


# -------------------------
# Delete event
# -------------------------

def test_delete_event_success(env, sample_event_dict):
    env.initialize(events=[sample_event_dict], now="2026-01-01 09:00:00")

    result = env.delete_event("evt_1")
    assert result["message"] == "Event deleted successfully."
    assert len(env.calendar.events) == 0


def test_delete_event_missing(env):
    result = env.delete_event("missing")
    assert "error" in result


# -------------------------
# Respond to event
# -------------------------

def test_respond_to_event_success(env, sample_event_dict):
    env.initialize(events=[sample_event_dict], now="2026-01-01 09:00:00")

    env.respond_to_event("evt_1", "ACCEPT")
    assert env.calendar.events[0].attending == "ACCEPT"


def test_respond_to_event_invalid_value(env, sample_event_dict):
    env.initialize(events=[sample_event_dict], now="2026-01-01 09:00:00")

    result = env.respond_to_event("evt_1", "YES")
    assert "error" in result


# -------------------------
# Current time
# -------------------------

def test_get_current_time(env):
    env.now = datetime(2026, 1, 1, 9, 0, 0)
    result = env.get_current_time()
    assert result["current_time"] == "2026-01-01 09:00:00"
