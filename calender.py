from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import uuid

@dataclass
class Event:
    id: str
    calendar_id: str
    summary: str
    start: str
    end: str
    description: str = ""
    location: str = ""
    attendees: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")


@dataclass
class Calendar:
    id: str
    summary: str
    primary: bool = False
    events: Dict[str, Event] = field(default_factory=dict)


class CalendarEnvironment:
    def __init__(self, timezone: str = "UTC"):
        self.timezone = timezone
        self.now = datetime.utcnow().isoformat() + "Z"
        self.calendars: Dict[str, Calendar] = {
            "primary": Calendar(
                id="primary",
                summary="Primary Calendar",
                primary=True
            )
        }

    def _get_calendar(self, calendar_id: str) -> Calendar:
        if calendar_id not in self.calendars:
            raise ValueError(f"CALENDAR_NOT_FOUND: {calendar_id}")
        return self.calendars[calendar_id]

    def _get_event(self, calendar_id: str, event_id: str) -> Event:
        calendar = self._get_calendar(calendar_id)
        if event_id not in calendar.events:
            raise ValueError(f"EVENT_NOT_FOUND: {event_id}")
        return calendar.events[event_id]

    def _error(self, code: str, message: str):
        return {
            "error": {
                "code": code,
                "message": message
            }
        }

    def list_calendars(self):
        return {
            "calendars": [
                {
                    "id": cal.id,
                    "summary": cal.summary,
                    "primary": cal.primary
                }
                for cal in self.calendars.values()
            ]
        }

    def list_events(self, calendar_id: str, time_min: Optional[str] = None, time_max: Optional[str] = None):
        calendar = self._get_calendar(calendar_id)

        events = list(calendar.events.values())

        if time_min:
            events = [e for e in events if e.end >= time_min]
        if time_max:
            events = [e for e in events if e.start <= time_max]

        return {
            "events": [
                {
                    "id": e.id,
                    "summary": e.summary,
                    "start": e.start,
                    "end": e.end
                }
                for e in events
            ]
        }

    def get_event(self, calendar_id: str, event_id: str):
        event = self._get_event(calendar_id, event_id)
        return {"event": event.__dict__}

    def create_event(
        self,
        calendar_id: str,
        summary: str,
        start: str,
        end: str,
        attendees: Optional[List[str]] = None,
        description: str = "",
        location: str = ""
    ):
        calendar = self._get_calendar(calendar_id)

        event_id = f"evt_{uuid.uuid4().hex[:8]}"
        attendee_map = {email: "needsAction" for email in (attendees or [])}

        event = Event(
            id=event_id,
            calendar_id=calendar_id,
            summary=summary,
            start=start,
            end=end,
            description=description,
            location=location,
            attendees=attendee_map
        )

        calendar.events[event_id] = event

        return {"event": event.__dict__}

    def update_event(self, calendar_id: str, event_id: str, **updates):
        event = self._get_event(calendar_id, event_id)

        for key, value in updates.items():
            if hasattr(event, key):
                setattr(event, key, value)

        event.updated_at = datetime.utcnow().isoformat() + "Z"
        return {"event": event.__dict__}

    def delete_event(self, calendar_id: str, event_id: str):
        calendar = self._get_calendar(calendar_id)

        if event_id not in calendar.events:
            return self._error("EVENT_NOT_FOUND", f"No event with id {event_id}")

        del calendar.events[event_id]
        return {"status": "deleted", "eventId": event_id}

    def respond_to_event(self, calendar_id: str, event_id: str, email: str, response_status: str):
        event = self._get_event(calendar_id, event_id)

        if email not in event.attendees:
            return self._error("ATTENDEE_NOT_FOUND", f"{email} not in attendees")

        event.attendees[email] = response_status
        event.updated_at = datetime.utcnow().isoformat() + "Z"

        return {
            "eventId": event_id,
            "email": email,
            "responseStatus": response_status
        }

    def get_freebusy(self, calendar_ids: List[str], time_min: str, time_max: str):
        busy = []

        for cid in calendar_ids:
            calendar = self._get_calendar(cid)
            for event in calendar.events.values():
                if not (event.end <= time_min or event.start >= time_max):
                    busy.append({
                        "calendarId": cid,
                        "start": event.start,
                        "end": event.end
                    })

        return {"busy": busy}

    def get_current_time(self):
        self.now = datetime.utcnow().isoformat() + "Z"
        return {
            "now": self.now,
            "timezone": self.timezone
        }


if __name__ == "__main__":
    env = CalendarEnvironment()

    env.create_event(
        calendar_id="primary",
        summary="Team Sync",
        start="2025-12-26T10:00:00Z",
        end="2025-12-26T11:00:00Z",
        attendees=["agent@local"]
    )

    print(env.list_events("primary"))
    print(env.get_freebusy(["primary"], "2025-12-26T09:00:00Z", "2025-12-26T17:00:00Z"))
