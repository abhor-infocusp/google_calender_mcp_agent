import json
import os
import pytest

from datetime import datetime

from environment.environment import CalendarEnvironment


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
JSON_CALENDAR_DIR = os.path.join(DATA_DIR, "json_calender")


def get_data_file(index: int) -> str:
    return os.path.join(JSON_CALENDAR_DIR, f"{index}.txt")


def get_earliest_start(file_path: str) -> str:
    """Derive a 'now' timestamp from the earliest event start in a calendar file."""
    with open(file_path) as f:
        data = json.load(f)

    earliest = None
    for day_events in data.values():
        for event in day_events:
            dt = datetime.fromisoformat(event["start"])
            if earliest is None or dt < earliest:
                earliest = dt

    return earliest.strftime("%Y-%m-%d %H:%M:%S")


# -------------------------
# Data loading
# -------------------------

class TestLoadJsonCalendar:

    def test_load_returns_list(self):
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        assert isinstance(events, list)
        assert len(events) > 0

    def test_events_have_required_fields(self):
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        required_fields = {"id", "summary", "start", "end", "attendees", "optional"}
        for event in events:
            assert required_fields.issubset(event.keys()), (
                f"Missing fields: {required_fields - event.keys()}"
            )

    def test_event_ids_are_unique(self):
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        ids = [e["id"] for e in events]
        assert len(ids) == len(set(ids))

    def test_attendees_are_dicts_with_user(self):
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        events_with_attendees = [e for e in events if e["attendees"]]
        assert len(events_with_attendees) > 0, "Expected at least one event with attendees"

        for event in events_with_attendees:
            for attendee in event["attendees"]:
                assert "user" in attendee
                assert "attending" in attendee
                assert "email" in attendee["user"]
                assert "name" in attendee["user"]
                assert "id" in attendee["user"]

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            CalendarEnvironment.load_json_calendar("/nonexistent/path.txt")

    def test_flattens_all_days(self):
        file_path = get_data_file(0)
        with open(file_path) as f:
            raw = json.load(f)

        expected_count = sum(len(events) for events in raw.values())
        loaded = CalendarEnvironment.load_json_calendar(file_path)
        assert len(loaded) == expected_count


# -------------------------
# Initialize with real data
# -------------------------

class TestInitializeWithData:

    def test_initialize_with_data_file_0(self):
        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        now = get_earliest_start(get_data_file(0))

        env.initialize(events=events, now=now)
        assert len(env.calendar.events) > 0

    def test_initialize_with_data_file_1(self):
        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(get_data_file(1))
        now = get_earliest_start(get_data_file(1))

        env.initialize(events=events, now=now)
        assert len(env.calendar.events) > 0

    def test_event_summaries_preserved(self):
        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        now = get_earliest_start(get_data_file(0))

        env.initialize(events=events, now=now)
        summaries = [e.summary for e in env.calendar.events]
        assert "Pre-flight Check & Briefing" in summaries

    def test_event_datetimes_parsed(self):
        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        now = get_earliest_start(get_data_file(0))

        env.initialize(events=events, now=now)
        for event in env.calendar.events:
            assert isinstance(event.start, datetime)
            assert isinstance(event.end, datetime)
            assert event.end > event.start

    def test_attendees_loaded(self):
        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        now = get_earliest_start(get_data_file(0))

        env.initialize(events=events, now=now)
        events_with_attendees = [e for e in env.calendar.events if e.attendees]
        assert len(events_with_attendees) > 0

        for event in events_with_attendees:
            for attendee in event.attendees:
                assert attendee.user.email
                assert attendee.attending in ["ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE"]


# -------------------------
# Operations on loaded data
# -------------------------

class TestOperationsWithData:

    @pytest.fixture
    def loaded_env(self):
        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(get_data_file(0))
        now = get_earliest_start(get_data_file(0))
        env.initialize(events=events, now=now)
        return env

    def test_list_events_returns_all(self, loaded_env):
        result = loaded_env.list_events()
        assert "events" in result
        assert len(result["events"]) == len(loaded_env.calendar.events)

    def test_list_events_filter_by_day(self, loaded_env):
        # Monday events for data file 0 are on 2024-01-01
        result = loaded_env.list_events(
            time_min="2024-01-01 00:00:00",
            time_max="2024-01-01 23:59:59",
        )
        assert "events" in result
        assert len(result["events"]) > 0
        # Should be fewer than total events
        assert len(result["events"]) < len(loaded_env.calendar.events)

    def test_get_event_by_id(self, loaded_env):
        event_id = loaded_env.calendar.events[0].id
        result = loaded_env.get_event(event_id)
        assert "event" in result

    def test_get_event_invalid_id(self, loaded_env):
        result = loaded_env.get_event("nonexistent_id")
        assert "error" in result

    def test_create_event_on_loaded_calendar(self, loaded_env):
        original_count = len(loaded_env.calendar.events)
        result = loaded_env.create_event(
            summary="New Meeting",
            start="2024-01-01 16:00:00",
            end="2024-01-01 17:00:00",
        )
        assert result["message"] == "Event created successfully."
        assert len(loaded_env.calendar.events) == original_count + 1

    def test_update_event_on_loaded_calendar(self, loaded_env):
        event_id = loaded_env.calendar.events[0].id
        result = loaded_env.update_event(
            event_id=event_id,
            updates={"summary": "Updated Event Name"},
        )
        assert result["message"] == "Event updated successfully."
        assert loaded_env.calendar.events[0].summary == "Updated Event Name"

    def test_delete_event_on_loaded_calendar(self, loaded_env):
        original_count = len(loaded_env.calendar.events)
        event_id = loaded_env.calendar.events[0].id
        result = loaded_env.delete_event(event_id)
        assert result["message"] == "Event deleted successfully."
        assert len(loaded_env.calendar.events) == original_count - 1

    def test_respond_to_event_on_loaded_calendar(self, loaded_env):
        event_id = loaded_env.calendar.events[0].id
        loaded_env.respond_to_event(event_id, "DECLINE")
        assert loaded_env.calendar.events[0].attending == "DECLINE"

    def test_get_current_time(self, loaded_env):
        result = loaded_env.get_current_time()
        assert "current_time" in result
        # Should match the earliest event start we derived
        datetime.strptime(result["current_time"], "%Y-%m-%d %H:%M:%S")


# -------------------------
# Multiple data files
# -------------------------

class TestMultipleDataFiles:

    @pytest.mark.parametrize("file_index", range(5))
    def test_load_and_initialize_data_files(self, file_index):
        """Test that the first 5 data files all load and initialize correctly."""
        file_path = get_data_file(file_index)
        if not os.path.exists(file_path):
            pytest.skip(f"Data file {file_index} not found")

        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(file_path)
        now = get_earliest_start(file_path)

        env.initialize(events=events, now=now)
        assert len(env.calendar.events) == len(events)

        # Verify list_events works
        result = env.list_events()
        assert "events" in result
        assert len(result["events"]) == len(events)

    @pytest.mark.parametrize("file_index", range(5))
    def test_all_events_have_valid_times(self, file_index):
        """Verify all events in loaded data have end > start."""
        file_path = get_data_file(file_index)
        if not os.path.exists(file_path):
            pytest.skip(f"Data file {file_index} not found")

        env = CalendarEnvironment()
        events = CalendarEnvironment.load_json_calendar(file_path)
        now = get_earliest_start(file_path)
        env.initialize(events=events, now=now)

        for event in env.calendar.events:
            assert event.end >= event.start, (
                f"Event '{event.summary}' has end < start"
            )
