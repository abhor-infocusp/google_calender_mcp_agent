import pytest

from calendar_agent.environment import CalendarEnvironment


def pytest_addoption(parser):
    parser.addoption(
        "--run-external",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.external (hit live Gemini etc).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "external: test requires external services (Gemini, vLLM, GPU)."
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-external"):
        return
    skip_external = pytest.mark.skip(reason="needs --run-external")
    for item in items:
        if "external" in item.keywords:
            item.add_marker(skip_external)


@pytest.fixture
def env():
    e = CalendarEnvironment()
    e.calendar.events = []
    e.now = None
    return e


@pytest.fixture
def populated_env():
    """CalendarEnvironment with 3 deterministic events on Mon/Tue/Wed."""
    e = CalendarEnvironment()
    e.calendar.events = []
    events = [
        {
            "id": "evt_mon",
            "summary": "Monday Sync",
            "start": "2026-01-05 10:00:00",
            "end": "2026-01-05 11:00:00",
            "attendees": [
                {
                    "user": {"id": "u1", "name": "alice", "email": "alice@example.com"},
                    "attending": "ACCEPT",
                }
            ],
            "optional": False,
        },
        {
            "id": "evt_tue",
            "summary": "Tuesday Review",
            "start": "2026-01-06 14:00:00",
            "end": "2026-01-06 15:00:00",
            "attendees": [],
            "optional": False,
        },
        {
            "id": "evt_wed_early",
            "summary": "Wednesday Early",
            "start": "2026-01-07 09:00:00",
            "end": "2026-01-07 09:30:00",
            "attendees": [],
            "optional": False,
        },
    ]
    e.initialize(events=events, now="2026-01-05 08:00:00")
    return e
