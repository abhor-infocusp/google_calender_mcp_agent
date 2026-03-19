import uuid

from datetime import datetime
from pydantic import BaseModel
from typing import List, Literal


class User(BaseModel):
    """Data model for storing user information"""
    id: str
    name: str
    email: str


class Attendee(BaseModel):
    """Data model for storing attendee information"""
    user: User
    attending: Literal["ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE"] = "ACCEPT"


class Event(BaseModel):
    """Data model for storing event information"""
    id: str
    summary: str
    start: datetime
    end: datetime
    description: str = ""
    attendees: List[Attendee] = []
    optional: bool
    attending: Literal["ACCEPT", "DECLINE", "MAYBE", "NO RESPONSE"] = "ACCEPT"


class Calendar(BaseModel):
    """Data model for storing calendar data."""
    id: str
    summary: str
    primary: bool = False
    events: List[Event] = []