import uuid

from datetime import datetime
from typing import List

from environment.models import Attendee, Calendar, Event, User

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
    
    def initialize(self, events: list[dict], users: list[dict], now: str) -> None:
        """Initialize the calendar with events."""
        try:
            self.calendar.events += [
                Event.model_validate(event) for event in events
            ]

            self.users += [
                User.model_validate(user) for user in users
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

    def _raise_error(self, type: str, message: str):
        return {
            "error": {
                "type": type,
                "message": message
            }
        }

    def list_calendars(self):
        raise NotImplementedError
        # TODO: Will be added when we start supporting multiple calendars.
        # return {
        #     "calendars": [
        #         {
        #             "id": cal.id,
        #             "summary": cal.summary,
        #             "primary": cal.primary
        #         }
        #         for cal in self.calendars.values()
        #     ]
        # }
    
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
            return {
                "events": [
                    e.model_dump_json(indent=2)
                    for e in events
                ]
            }
        except ValueError as e:
            return self._raise_error("ValueError", str(e))
        except Exception as e:
            return self._raise_error("Exception", str(e))

    def get_event(self, event_id: str):
        try:
            event = self._get_event(event_id)
            return {"event": event.model_dump_json(indent=2)}
        except ValueError as e:
            self._raise_error("ValueError", str(e))
        except Exception as e:
            self._raise_error("Exception", str(e))

    def create_event(
        self,
        summary: str,
        start: str,
        end: str,
        attendees: list[dict] = [],
        description: str = "",
        optional=False,
        organizer=None
    ):
        """Create a calendar event."""        
        try:
            start = self._parse_datetime_str(start)
            end = self._parse_datetime_str(end)
            attendees = [Attendee.model_validate_json(a) for a in attendees]
            organizer = User.model_validate_json(organizer) if organizer else self.user

            event_id = f"evt_{uuid.uuid4().hex}"
            event = Event(
                id=event_id,
                summary=summary,
                start=start,
                end=end,
                attendees=attendees,
                description=description,
                optional=optional,
                organizer=organizer
            )

            self.calendar.events.append(event)

            return {
                "message": "Event created successfully.", 
                "event": event.__dict__
            }
        except ValueError as e:
            self._raise_error("ValueError", str(e))
        except Exception as e:
            self._raise_error("Exception", str(e))

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
                if field == "end":
                    event.end = self._parse_datetime_str(value)
                if field == "attendees":
                    event.attendees = [Attendee.model_validate_json(v) for v in value]
                if field == "summary":
                    event.summary = value
                if field == "description":
                    event.description = value
                if field == "optional":
                    event.optional = value
                if field == "organizer":
                    event.optional = value

            return {
                "message": "Event updated successfully.", 
                "event": event.__dict__
            }
        except ValueError as e:
            self._raise_error("ValueError", str(e))
        except Exception as e:
            self._raise_error("Exception", str(e))

    def delete_event(self, event_id: str):
        """Delete a particular event."""
        try:
            event = self._get_event(event_id)
            self.calendar.events.remove(event)
            return {
                "message": "Event deleted successfully.",
                "event": event.__dict__
            }
        except ValueError as e:
            self._raise_error("ValueError", str(e))
        except Exception as e:
            self._raise_error("Exception", str(e))

    def respond_to_event(self, event_id: str, attending: str):
        """Respond to an event."""
        try:
            event = self._get_event(event_id)
            if attending not in ["ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE"]:
                raise ValueError(f"Invalid value for `attending` field, expected ACCEPT, DECLINE, MAYBE, NO RESPONSE got {attending}.")
            event.attending = attending
        except ValueError as e:
            self._raise_error("ValueError", str(e))
        except Exception as e:
            self._raise_error("Exception", str(e))

    # TODO: Improve Implementation
    # def get_freebusy(self, time_min: str, time_max: str):
    #     """Get free and busy slots for the user."""
    #     try:
    #         time_min = self._parse_datetime_str(time_min)
    #         time_max = self._parse_datetime_str(time_max)

    #         busy = []

    #         for event in self.calendar.events:
    #             if not(event.end <= time_min or event.start >= time_max):
    #                 busy.append()
    #     except ValueError as e:
    #         self._raise_error("ValueError", str(e))
    #     except Exception as e:
    #         self._raise_error("Exception", str(e))

    def get_current_time(self):
        return {
            "current_time": self.now.strftime("%Y-%m-%d %H:%M:%S")
        }