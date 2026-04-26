"""Opt-in tests that call live Gemini. Run with: pytest --run-external"""
import pytest

from calendar_agent.evaluation import EVAL_SYSTEM_PROMPT, evaluate_trajectory


pytestmark = pytest.mark.external


@pytest.fixture(scope="module")
def gemini_model():
    # Per project memory: NEVER use Pro models. Always gemini-2.0-flash-001.
    from vertexai.generative_models import GenerativeModel
    return GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])


def test_evaluate_trajectory_returns_valid_verdict(gemini_model):
    before = {
        "Monday": [
            {
                "summary": "Standup",
                "start": "2026-01-05 09:00:00",
                "end": "2026-01-05 09:30:00",
                "attendees": [],
                "attending": "ACCEPT",
            }
        ]
    }
    after = {
        "Monday": [
            {
                "summary": "Standup",
                "start": "2026-01-05 09:00:00",
                "end": "2026-01-05 09:30:00",
                "attendees": [],
                "attending": "ACCEPT",
            },
            {
                "summary": "Lunch",
                "start": "2026-01-05 12:00:00",
                "end": "2026-01-05 13:00:00",
                "attendees": [],
                "attending": "ACCEPT",
            },
        ]
    }
    verdict, reasoning = evaluate_trajectory(
        eval_model=gemini_model,
        query="Add a lunch event Monday at noon for one hour.",
        final_output="Created 'Lunch' on Monday 12:00-13:00.",
        expected="Create a Lunch event Monday 12:00-13:00.",
        before_days=before,
        after_days=after,
    )
    assert verdict in ("Correct", "Incorrect")
    assert isinstance(reasoning, str) and reasoning.strip()
