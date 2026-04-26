from string import Template

import pytest

from calendar_agent import prompts


ALL_TEMPLATES = ["persona_prompt", "architect_prompt", "jsonizer_prompt", "query_prompt"]


@pytest.mark.parametrize("name", ALL_TEMPLATES)
def test_is_template(name):
    assert isinstance(getattr(prompts, name), Template)


def test_persona_prompt_substitutes():
    out = prompts.persona_prompt.substitute(profession="hospital administrator")
    assert "hospital administrator" in out


def test_architect_prompt_substitutes():
    out = prompts.architect_prompt.substitute(persona="A senior administrator.")
    assert "A senior administrator." in out


def test_jsonizer_prompt_substitutes():
    out = prompts.jsonizer_prompt.substitute(
        calender_text="Monday: 10:00-11:00 Standup",
        monday_date="2026-01-05",
    )
    assert "2026-01-05" in out
    assert "Monday: 10:00-11:00 Standup" in out


def test_query_prompt_substitutes():
    out = prompts.query_prompt.substitute(
        calender_text="<calendar>",
        monday_date="2026-01-05",
    )
    assert "<calendar>" in out
    assert "2026-01-05" in out


def test_jsonizer_prompt_missing_placeholder_raises():
    with pytest.raises(KeyError):
        prompts.jsonizer_prompt.substitute(calender_text="x")


@pytest.mark.parametrize("name,expected", [
    ("persona_prompt", {"profession"}),
    ("architect_prompt", {"persona"}),
    ("jsonizer_prompt", {"calender_text", "monday_date"}),
    ("query_prompt", {"calender_text", "monday_date"}),
])
def test_template_identifiers(name, expected):
    tpl = getattr(prompts, name)
    assert set(tpl.get_identifiers()) == expected
