# Project Documentation

## Project Overview

This is a Google Calendar MCP (Model Context Protocol) agent project. It provides a simulated calendar environment for testing and development purposes, with data models for users, events, attendees, and calendars. The project also includes a synthetic data generation prompt for creating realistic calendar event data.

---

## Directory Structure

### Root Directory

#### `main.py`
**Description:** Entry point for the project. A simple script that prints a greeting message.

**Module-level Functions:**
- `main()` - Prints "Hello from google-calender-mcp-agent!" to the console.

---

### `environment/` Directory

#### `models.py`
**Description:** Pydantic data models for the calendar system. Defines the core data structures for users, attendees, events, and calendars.

**Classes:**
- `User`
  - **Description:** Data model for storing user information
  - **Fields:**
    - `id: str` - Unique user identifier
    - `name: str` - User's full name
    - `email: str` - User's email address

- `Attendee`
  - **Description:** Data model for storing attendee information
  - **Fields:**
    - `user: User` - The user who is an attendee
    - `attending: Literal["ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE"]` - Attendance status (default: "ACCEPT")

- `Event`
  - **Description:** Data model for storing event information
  - **Fields:**
    - `id: str` - Unique event identifier
    - `summary: str` - Event title/summary
    - `start: datetime` - Event start time
    - `end: datetime` - Event end time
    - `description: str` - Event description (default: "")
    - `attendees: List[Attendee]` - List of attendees (default: [])
    - `organizer: User` - Event organizer
    - `optional: bool` - Whether the event is optional
    - `attending: Literal["ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE"]` - Attendance status (default: "ACCEPT")

- `Calendar`
  - **Description:** Data model for storing calendar data
  - **Fields:**
    - `id: str` - Calendar identifier
    - `summary: str` - Calendar name/summary
    - `primary: bool` - Whether this is the primary calendar (default: False)
    - `events: List[Event]` - List of events in the calendar (default: [])

---

#### `environment.py`
**Description:** Main calendar environment class that provides an interface for managing calendar events. Supports, listing, updating, deleting events creating, and responding to event invitations.

**Classes:**
- `CalendarEnvironment`
  - **Description:** The primary calendar environment class

  **Constructor:**
  - `__init__()` - Sets up the calendar with a primary calendar and default user

  **Methods:**
  - `initialize(events: list[dict], users: list[dict], now: str) -> None`
    - **Description:** Initialize the calendar with events and users
    - **Parameters:**
      - `events: list[dict]` - List of event dictionaries
      - `users: list[dict]` - List of user dictionaries
      - `now: str` - Current time in "YYYY-MM-DD HH:MM:SS" format

  - `_parse_datetime_str(dt_str: str) -> datetime`
    - **Description:** Parse a datetime string into a datetime object
    - **Parameters:**
      - `dt_str: str` - Datetime string in "YYYY-MM-DD HH:MM:SS" format
    - **Returns:** datetime object

  - `_get_event(event_id: str) -> Event`
    - **Description:** Internal method to retrieve an event by ID
    - **Parameters:**
      - `event_id: str` - The event identifier
    - **Returns:** Event object
    - **Raises:** ValueError if event not found

  - `_raise_error(type: str, message: str) -> dict`
    - **Description:** Internal method to create error response dictionaries

  - `list_calendars()`
    - **Description:** Not implemented. Placeholder for listing multiple calendars.

  - `_list_events(time_min: datetime = None, time_max: datetime = None) -> list`
    - **Description:** Internal function for listing all events within a provided time range

  - `list_events(time_min: str = None, time_max: str = None) -> dict`
    - **Description:** Function exposed to the LLM for listing events
    - **Parameters:**
      - `time_min: str` - Minimum time value to filter events (format: "YYYY-MM-DD HH:MM:SS")
      - `time_max: str` - Maximum time value to filter events (format: "YYYY-MM-DD HH:MM:SS")

  - `get_event(event_id: str) -> dict`
    - **Description:** Get a specific event by ID

  - `create_event(summary: str, start: str, end: str, attendees: list[dict] = [], description: str = "", optional=False, organizer=None) -> dict`
    - **Description:** Create a calendar event
    - **Parameters:**
      - `summary: str` - Event title
      - `start: str` - Start time (format: "YYYY-MM-DD HH:MM:SS")
      - `end: str` - End time (format: "YYYY-MM-DD HH:MM:SS")
      - `attendees: list[dict]` - List of attendee dictionaries
      - `description: str` - Event description
      - `optional: bool` - Whether event is optional
      - `organizer: dict` - Organizer user dictionary

  - `update_event(event_id: str, updates: dict) -> dict`
    - **Description:** Update an existing event
    - **Parameters:**
      - `event_id: str` - The event identifier
      - `updates: dict` - Dictionary of fields to update

  - `delete_event(event_id: str) -> dict`
    - **Description:** Delete a particular event

  - `respond_to_event(event_id: str, attending: str)`
    - **Description:** Respond to an event invitation
    - **Parameters:**
      - `event_id: str` - The event identifier
      - `attending: str` - Response status ("ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE")

  - `get_current_time() -> dict`
    - **Description:** Get the current simulated time

---

### `tests/` Directory

#### `test_calendar_env.py`
**Description:** Pytest test suite for the CalendarEnvironment class. Contains comprehensive tests for initialization, datetime parsing, listing events, getting events, creating events, updating events, deleting events, responding to events, and getting current time.

**Fixtures:**
- `env` - Creates a CalendarEnvironment test fixture with empty events and users
- `sample_event_dict` - Sample event dictionary for testing
- `sample_user_dict` - Sample user dictionary for testing

**Test Functions:**
- `test_initialize_success` - Tests successful environment initialization
- `test_initialize_invalid_datetime_raises` - Tests initialization with invalid datetime
- `test_parse_datetime_success` - Tests successful datetime parsing
- `test_parse_datetime_invalid_format` - Tests datetime parsing with invalid format
- `test_list_events_no_filters` - Tests listing events without filters
- `test_list_events_with_time_min` - Tests listing events with time_min filter
- `test_list_events_with_time_max` - Tests listing events with time_max filter
- `test_list_events_invalid_time` - Tests listing events with invalid time input
- `test_get_event_success` - Tests getting an event by ID
- `test_get_event_not_found` - Tests getting a non-existent event
- `test_create_event_success` - Tests creating an event successfully
- `test_create_event_invalid_start` - Tests creating an event with invalid start time
- `test_update_event_summary` - Tests updating an event's summary
- `test_update_event_invalid_id` - Tests updating a non-existent event
- `test_delete_event_success` - Tests deleting an event successfully
- `test_delete_event_missing` - Tests deleting a non-existent event
- `test_respond_to_event_success` - Tests responding to an event
- `test_respond_to_event_invalid_value` - Tests responding with invalid value
- `test_get_current_time` - Tests getting the current time

---

### `data_generation/` Directory

#### `prompt_v1.txt`
**Description:** A prompt template for generating synthetic calendar data. Contains instructions for creating realistic calendar events for a fictional person over one week.

**Prompt Structure:**
- **Phase 1: Persona Creation** - Guidelines for creating a fictional persona with identity, responsibilities, constraints, and relevant people
- **Phase 2: Calendar Blueprint** - Instructions for designing a realistic weekly schedule with event density and text-based schedule
- **Phase 3: JSON Conversion** - Instructions for converting the calendar blueprint into valid JSON conforming to the Event schema

---

### Configuration Files

#### `pyproject.toml`
**Description:** Project configuration file for uv package manager

**Project Metadata:**
- Name: `google-calender-mcp-agent`
- Version: `0.1.0`
- Python Requirement: `>=3.12`
- Dependencies:
  - `pydantic>=2.12.5`
  - `pytest>=9.0.2`

#### `README.md`
**Description:** Project setup instructions. Contains steps for installing uv, initializing the project, setting up the virtual environment, and running tests.

---

## Module-Level Variables

### `environment/environment.py`
- `self.calendar` - Calendar instance (initialized in `__init__`)
- `self.user` - Default User instance (initialized in `__init__`)
- `self.users` - List of users (populated in `initialize`)
- `self.now` - Current datetime (populated in `initialize`)

---

## Known Issues (from test comments)

- `get_event` with not found event returns None instead of error
- `create_event` with invalid start returns None instead of error
- `update_event` with invalid ID returns None instead of error
- `delete_event` with missing event returns None
- `respond_to_event` with invalid value returns None instead of error
