import json
import uuid

from datetime import datetime
from typing import List

from calendar_agent.environment.models import Attendee, Calendar, Event, User

class CalendarEnvironment:
    """The primary calendar environment."""

    def __init__(self):
        """Setup the calendar."""
        self.calendar = Calendar(
            id="primary",
            summary="Primary Calendar",
            primary=True
        )

        self.user = User(
            id="1",
            name="John Doe",
            email="john.doe@example.com"
        )

    @staticmethod
    def _format_summary(event: "Event") -> str:
        """One-line summary: id: <id> | <summary> — <Day> <HH:MM>-<HH:MM>"""
        day = event.start.strftime("%a")
        start_t = event.start.strftime("%H:%M")
        end_t = event.end.strftime("%H:%M")
        return f"id: {event.id} | {event.summary} — {day} {start_t}-{end_t}"

    @staticmethod
    def _format_detail(event: "Event") -> str:
        """Multi-line detail block with all fields including RSVP."""
        day_date = event.start.strftime("%a %b %d")
        start_t = event.start.strftime("%H:%M")
        end_t = event.end.strftime("%H:%M")
        lines = [
            event.summary,
            f"  ID: {event.id}",
            f"  Time: {day_date}, {start_t} - {end_t}",
        ]
        if event.description:
            lines.append(f"  Description: {event.description}")
        if event.attendees:
            emails = ", ".join(a.user.email for a in event.attendees)
            lines.append(f"  Attendees: {emails}")
        lines.append(f"  RSVP: {event.attending}")
        return "\n".join(lines)
    
    @staticmethod
    def load_json_calendar(file_path: str) -> list[dict]:
        """Load a JSON calendar file and transform it into event dicts for initialize().

        The JSON files are keyed by day-of-week with events containing summary,
        start, end, and attendees (as email strings). This method flattens
        the structure and adds required fields (id, optional, attendee objects).
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        events = []
        for day, day_events in data.items():
            for event in day_events:
                attendees = [
                    {
                        "user": {
                            "id": f"user_{uuid.uuid4().hex[:8]}",
                            "name": email.split("@")[0],
                            "email": email,
                        },
                        "attending": "ACCEPT",
                    }
                    for email in event.get("attendees", [])
                ]

                events.append({
                    "id": f"evt_{uuid.uuid4().hex}",
                    "summary": event["summary"],
                    "start": event["start"],
                    "end": event["end"],
                    "attendees": attendees,
                    "optional": False,
                })

        return events

    def initialize(self, events: list[dict], now: str) -> None:
        """Initialize the calendar with events."""
        try:
            self.calendar.events += [
                Event.model_validate(event) for event in events
            ]

            self.now = datetime.strptime(now, "%Y-%m-%d %H:%M:%S")
        except ValueError as e:
            raise Exception("Event data should be according to schema.")
    
    def _parse_datetime_str(self, dt_str: str) -> datetime:
        try:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            raise Exception(
                f"Expected date time input to be in YYYY-MM-DD HH:MM:SS format, got {dt_str}."
            )

    def _get_event(self, event_id: str) -> Event:
        for e in self.calendar.events:
            if e.id == event_id:
                return e
        raise ValueError(f"Could not find event with id {event_id}")

    def _list_events(self, time_min: datetime = None, time_max: datetime = None):
        """Internal function for listing all events within a provided time range."""
        events = [e for e in self.calendar.events]

        if time_min:
            events = [e for e in events if e.end >= time_min]
        
        if time_max:
            events = [e for e in events if e.start <= time_max]
        
        return events

    def list_events(self, time_min: str = None, time_max: str = None) -> dict:
        """Function that will be exposed to the LLM.

        Args:
            time_min: Minimum time value to filter events.
            time_max: Maximum time value to filter events.
        """

        try:
            time_min = self._parse_datetime_str(time_min) if time_min else time_min
            time_max = self._parse_datetime_str(time_max) if time_max else time_max
            events = self._list_events(time_min, time_max)
            lines = [f"Found {len(events)} events:"]
            for e in events:
                lines.append(self._format_summary(e))
            return "\n".join(lines)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def get_event(self, event_id: str):
        try:
            event = self._get_event(event_id)
            return self._format_detail(event)
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def create_event(
        self,
        summary: str,
        start: str,
        end: str,
        attendees: list[dict] | None = None,
        description: str = "",
        optional=False,
    ):
        """Create a calendar event."""
        try:
            start = self._parse_datetime_str(start)
            end = self._parse_datetime_str(end)
            attendees = [Attendee.model_validate_json(a) for a in (attendees or [])]

            event_id = f"evt_{uuid.uuid4().hex}"
            event = Event(
                id=event_id,
                summary=summary,
                start=start,
                end=end,
                attendees=attendees,
                description=description,
                optional=optional,
            )

            self.calendar.events.append(event)

            return f"Event created successfully.\n{self._format_detail(event)}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def update_event(
        self, 
        event_id: str,
        updates: dict,
    ):
        try:
            event = self._get_event(event_id)

            for field, value in updates.items():
                if field == "start":
                    event.start = self._parse_datetime_str(value)
                elif field == "end":
                    event.end = self._parse_datetime_str(value)
                elif field == "attendees":
                    event.attendees = [Attendee.model_validate_json(v) for v in value]
                elif field == "summary":
                    event.summary = value
                elif field == "description":
                    event.description = value
                elif field == "optional":
                    event.optional = value
            return f"Event updated successfully.\n{self._format_detail(event)}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def delete_event(self, event_id: str):
        """Delete a particular event."""
        try:
            event = self._get_event(event_id)
            summary = self._format_summary(event)
            self.calendar.events.remove(event)
            return f"Event deleted: {summary}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def respond_to_event(self, event_id: str, attending: str):
        """Respond to an event."""
        try:
            event = self._get_event(event_id)
            if attending not in ["ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE"]:
                raise ValueError(f"Invalid value for `attending` field, expected ACCEPT, DECLINE, MAYBE, NO RESPONSE got {attending}.")
            event.attending = attending
            return f"RSVP updated to {attending}.\n{self._format_detail(event)}"
        except ValueError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    def get_current_time(self):
        return f"{self.now.strftime('%Y-%m-%d %H:%M:%S')} | {self.now.strftime('%A')}"