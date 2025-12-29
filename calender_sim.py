import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union, Any
from dataclasses import dataclass, field, asdict

# --- 1. Data Models (Schema Simulation) ---

@dataclass
class CalendarTime:
    """Represents the start/end time structure."""
    dateTime: Optional[str] = None  # ISO 8601
    date: Optional[str] = None      # YYYY-MM-DD
    timeZone: Optional[str] = None

@dataclass
class Attendee:
    """Represents an event attendee."""
    email: str
    responseStatus: str = "needsAction"  # needsAction, declined, tentative, accepted
    displayName: Optional[str] = None
    comment: Optional[str] = None

@dataclass
class Event:
    """Represents a Google Calendar Event resource."""
    id: str
    summary: str
    start: CalendarTime
    end: CalendarTime
    status: str = "confirmed"
    description: Optional[str] = None
    location: Optional[str] = None
    htmlLink: Optional[str] = None
    attendees: List[Attendee] = field(default_factory=list)
    created: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    updated: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

@dataclass
class Calendar:
    """Represents a Calendar resource."""
    id: str
    summary: str
    timeZone: str = "UTC"
    description: str = ""
    accessRole: str = "owner"
    backgroundColor: str = "#9a9cff"

# --- 2. The Mock Database (State Management) ---

class MockDatabase:
    """Simulates the Google Cloud backend storage."""
    def __init__(self):
        # Structure: { account_name: { calendar_id: CalendarObj } }
        self.calendars: Dict[str, Dict[str, Calendar]] = {}
        # Structure: { calendar_id: { event_id: EventObj } }
        self.events: Dict[str, Dict[str, Event]] = {}
        
        # Initialize with a default setup
        self.init_defaults()

    def init_defaults(self):
        self.add_account("default")
        self.create_calendar("default", "primary", "My Primary Calendar")

    def add_account(self, account: str):
        if account not in self.calendars:
            self.calendars[account] = {}

    def create_calendar(self, account: str, cal_id: str, summary: str):
        if cal_id not in self.events:
            self.events[cal_id] = {}
        self.calendars[account][cal_id] = Calendar(id=cal_id, summary=summary)

# --- 3. The MCP Tool Simulator ---

class GoogleCalendarMCPSimulator:
    def __init__(self):
        self.db = MockDatabase()
        self.current_account = "default"

    def _get_time_obj(self, time_input: Union[str, dict]) -> CalendarTime:
        """Helper to parse input time strings or dicts into CalendarTime objects."""
        if isinstance(time_input, dict):
            return CalendarTime(**time_input)
        # Assume ISO string if just a string is passed
        return CalendarTime(dateTime=time_input)

    # --- Tool: manage-accounts ---
    def manage_accounts(self, action: str, accountId: Optional[str] = None):
        """Add, remove, or list accounts."""
        if action == "list":
            return list(self.db.calendars.keys())
        elif action == "add" and accountId:
            self.db.add_account(accountId)
            return f"Account '{accountId}' added."
        elif action == "remove" and accountId:
            if accountId in self.db.calendars:
                del self.db.calendars[accountId]
                return f"Account '{accountId}' removed."
            return "Account not found."
        return "Invalid parameters."

    # --- Tool: list-calendars ---
    def list_calendars(self, account: Optional[str] = None):
        """Lists calendars for a specific account or all accounts."""
        accounts_to_search = [account] if account else self.db.calendars.keys()
        result = []
        
        for acc in accounts_to_search:
            if acc in self.db.calendars:
                result.extend([asdict(cal) for cal in self.db.calendars[acc].values()])
        return result

    # --- Tool: list-events ---
    def list_events(self, calendarId: str, timeMin: Optional[str] = None, timeMax: Optional[str] = None, maxResults: int = 10):
        """Lists events with basic time filtering."""
        if calendarId not in self.db.events:
            return {"error": f"Calendar {calendarId} not found."}

        all_events = list(self.db.events[calendarId].values())
        
        # Simple string comparison for ISO dates (simulation logic)
        filtered = []
        for e in all_events:
            # Skip if event ends before timeMin
            if timeMin and e.end.dateTime and e.end.dateTime < timeMin:
                continue
            # Skip if event starts after timeMax
            if timeMax and e.start.dateTime and e.start.dateTime > timeMax:
                continue
            filtered.append(asdict(e))

        return filtered[:maxResults]

    # --- Tool: create-event ---
    def create_event(self, calendarId: str, summary: str, start: str, end: str, description: str = "", location: str = ""):
        """Creates a new event."""
        if calendarId not in self.db.events:
            return {"error": f"Calendar {calendarId} not found."}

        event_id = str(uuid.uuid4().hex[:10])
        new_event = Event(
            id=event_id,
            summary=summary,
            description=description,
            location=location,
            start=self._get_time_obj(start),
            end=self._get_time_obj(end),
            htmlLink=f"https://calendar.google.com/mock/{event_id}"
        )
        
        self.db.events[calendarId][event_id] = new_event
        return asdict(new_event)

    # --- Tool: update-event ---
    def update_event(self, calendarId: str, eventId: str, **kwargs):
        """Updates specific fields of an event."""
        if calendarId not in self.db.events or eventId not in self.db.events[calendarId]:
            return {"error": "Event not found."}

        event = self.db.events[calendarId][eventId]
        
        # Update fields dynamically
        for key, value in kwargs.items():
            if hasattr(event, key):
                if key in ['start', 'end']:
                    setattr(event, key, self._get_time_obj(value))
                else:
                    setattr(event, key, value)
        
        event.updated = datetime.utcnow().isoformat() + "Z"
        return asdict(event)

    # --- Tool: delete-event ---
    def delete_event(self, calendarId: str, eventId: str):
        if calendarId in self.db.events and eventId in self.db.events[calendarId]:
            del self.db.events[calendarId][eventId]
            return {"status": "success", "message": "Event deleted"}
        return {"error": "Event not found."}

    # --- Tool: search-events ---
    def search_events(self, query: str, calendarId: Optional[str] = None):
        """Case-insensitive search in summary and description."""
        results = []
        cals_to_search = [calendarId] if calendarId else self.db.events.keys()

        for cal_id in cals_to_search:
            if cal_id in self.db.events:
                for event in self.db.events[cal_id].values():
                    content = (event.summary + (event.description or "")).lower()
                    if query.lower() in content:
                        results.append(asdict(event))
        return results

    # --- Tool: get-event ---
    def get_event(self, calendarId: str, eventId: str):
        if calendarId in self.db.events and eventId in self.db.events[calendarId]:
            return asdict(self.db.events[calendarId][eventId])
        return {"error": "Event not found."}

    # --- Tool: respond-to-event ---
    def respond_to_event(self, calendarId: str, eventId: str, response: str, comment: Optional[str] = None):
        """Simulates responding as the primary user."""
        if calendarId in self.db.events and eventId in self.db.events[calendarId]:
            event = self.db.events[calendarId][eventId]
            # In a real scenario, we find the attendee matching 'me'.
            # Here, we just mock adding/updating a 'self' attendee.
            me = Attendee(email="me@example.com", responseStatus=response, comment=comment)
            event.attendees.append(me)
            return {"status": "success", "response": response}
        return {"error": "Event not found."}

    # --- Tool: get-freebusy ---
    def get_freebusy(self, timeMin: str, timeMax: str, items: List[Dict[str, str]]):
        """Calculates busy periods."""
        response = {"calendars": {}}
        
        for item in items:
            cal_id = item['id']
            busy_slots = []
            if cal_id in self.db.events:
                for event in self.db.events[cal_id].values():
                    # Simplified logic: If event exists, it's busy (ignoring exact overlaps for brevity)
                    if event.start.dateTime and event.end.dateTime:
                         busy_slots.append({
                             "start": event.start.dateTime,
                             "end": event.end.dateTime
                         })
            response["calendars"][cal_id] = {"busy": busy_slots}
        
        return response

    # --- Tool: get-current-time ---
    def get_current_time(self):
        return datetime.now().isoformat()

    # --- Tool: list-colors ---
    def list_colors(self):
        return {
            "calendar": {"1": {"background": "#ac725e"}, "2": {"background": "#d06b64"}},
            "event": {"1": {"background": "#a4bdfc"}, "2": {"background": "#7ae7bf"}}
        }

# --- 4. Simulation Execution (Test Drive) ---

if __name__ == "__main__":
    sim = GoogleCalendarMCPSimulator()
    
    print("--- 1. List Initial Calendars ---")
    print(json.dumps(sim.list_calendars(), indent=2))

    print("\n--- 2. Create an Event ---")
    event_payload = {
        "calendarId": "primary",
        "summary": "Meeting with MCP Team",
        "start": "2023-11-01T10:00:00Z",
        "end": "2023-11-01T11:00:00Z",
        "description": "Discussing simulation architecture",
        "location": "Virtual"
    }
    created_event = sim.create_event(**event_payload)
    print(json.dumps(created_event, indent=2))
    
    # Store ID for future operations
    evt_id = created_event['id']

    print(f"\n--- 3. Update Event (Change Summary) ---")
    updated = sim.update_event(
        calendarId="primary", 
        eventId=evt_id, 
        summary="Urgent: Meeting with MCP Team"
    )
    print(f"New Summary: {updated['summary']}")

    print("\n--- 4. Search Events ---")
    search_res = sim.search_events(query="Urgent")
    print(f"Found {len(search_res)} event(s) matching 'Urgent'")

    print("\n--- 5. Get FreeBusy ---")
    fb = sim.get_freebusy(
        timeMin="2023-11-01T00:00:00Z", 
        timeMax="2023-11-02T00:00:00Z", 
        items=[{"id": "primary"}]
    )
    print(json.dumps(fb, indent=2))