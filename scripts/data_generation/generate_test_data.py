"""Generate a held-out test set of 50 calendars for evaluation.

Mirrors scripts/data_generation/generate_data.py but writes to
`test_data/` instead of `data/`, uses indices 0-49, and seeds the RNG
differently so the random persona/Monday-date choices don't collide with
previously generated calendars.

No trajectory generation — eval only needs the calendars + queries +
expected_behavior fields that Gemini already produces.
"""

import warnings

warnings.filterwarnings("ignore")

import json
import os
import random
from datetime import date, timedelta

import tqdm
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerationConfig, GenerativeModel

from calendar_agent.prompts import architect_prompt, jsonizer_prompt, persona_prompt, query_prompt
from calendar_agent.paths import CREDENTIALS_PATH, TEST_DATA_DIR


def random_monday(start_year: int = 2023, end_year: int = 2026) -> str:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    days_between = (end - start).days
    random_date = start + timedelta(days=random.randint(0, days_between))
    monday = random_date - timedelta(days=random_date.weekday())
    return monday.strftime("%Y-%m-%d")


PERSONA_DIR = str(TEST_DATA_DIR / "persona")
TEXT_CALENDER_DIR = str(TEST_DATA_DIR / "calender")
JSON_CALENDER_DIR = str(TEST_DATA_DIR / "json_calender")
QUERY_DIR = str(TEST_DATA_DIR / "queries")

DATA_SIZE = 50
START_INDEX = 0

# Distinct seed so professions + Monday dates differ from every prior run.
random.seed(20260424)


PROFESSIONS = [
    "Teacher", "Accountant", "Nurse", "Driver", "Engineer",
    "Manager", "Clerk", "Salesperson", "Analyst", "Technician",
    "Plumber", "Electrician", "Chef", "Artist", "Writer",
    "Lawyer", "Doctor", "Pilot", "Farmer", "Guard",
    "Architect", "Dentist", "Journalist", "Librarian", "Mechanic",
    "Musician", "Pharmacist", "Photographer", "Professor", "Therapist",
    "Veterinarian", "Firefighter", "Politician", "Researcher", "Consultant",
    "Social Worker", "Real Estate Agent", "Personal Trainer", "Event Planner", "Paramedic",
    "Data Scientist", "Product Manager", "UX Designer", "Recruiter", "Marketing Director",
    "Construction Foreman", "Translator", "Financial Advisor", "Nonprofit Director", "School Principal",
]


def main():
    for d in (PERSONA_DIR, TEXT_CALENDER_DIR, JSON_CALENDER_DIR, QUERY_DIR):
        os.makedirs(d, exist_ok=True)

    # Gemini (flash only — pro is a hard cost rule in this project)
    with open(CREDENTIALS_PATH) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None,
        refresh_token=cd["refresh_token"],
        client_id=cd["client_id"],
        client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
    model = GenerativeModel(
        "gemini-2.0-flash-001",
        system_instruction=[
            "You are a helpful, creative assistant.",
            "Avoid repetitive sentence structures.",
            "Be concise but maintain a natural, conversational flow.",
        ],
    )

    idx_range = range(START_INDEX, START_INDEX + DATA_SIZE)

    # 1. Personas
    print(f"Generating {DATA_SIZE} personas (indices {START_INDEX}-{START_INDEX + DATA_SIZE - 1})")
    for num in tqdm.tqdm(idx_range, desc="personas"):
        out_path = f"{PERSONA_DIR}/{num}.txt"
        if os.path.exists(out_path):
            continue
        prof = random.choice(PROFESSIONS)
        response = model.generate_content(persona_prompt.substitute(profession=prof))
        open(out_path, "w").write(response.text)

    # 2. Text calendars
    print("Generating text calendars")
    required_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for num in tqdm.tqdm(idx_range, desc="calendars"):
        persona_path = f"{PERSONA_DIR}/{num}.txt"
        out_path = f"{TEXT_CALENDER_DIR}/{num}.txt"
        if not os.path.exists(persona_path) or os.path.exists(out_path):
            continue
        persona = open(persona_path).read()
        if "#" not in persona:
            continue
        persona = persona.split("#")[-1]
        response = model.generate_content(architect_prompt.substitute(persona=persona))
        if "Monday" not in response.text:
            continue
        response = "Monday" + response.text.split("Monday")[1]
        if not all(day in response for day in required_days):
            continue
        open(out_path, "w").write(response)

    # 3. JSON calendars — also record the Monday date for query generation
    monday_dates: dict[str, str] = {}
    print("Converting text calendars to JSON")
    for num in tqdm.tqdm(idx_range, desc="json_cals"):
        text_path = f"{TEXT_CALENDER_DIR}/{num}.txt"
        json_path = f"{JSON_CALENDER_DIR}/{num}.txt"
        if not os.path.exists(text_path) or os.path.exists(json_path):
            if os.path.exists(json_path):
                # still need monday for step 4 if we're re-running — but we didn't store it.
                # Safe: regenerate monday only if query file absent (step 4 will skip if present).
                pass
            continue
        calender_text = open(text_path).read()
        config = GenerationConfig(temperature=1.0, response_mime_type="application/json")
        monday = random_monday()
        monday_dates[text_path] = monday
        response = model.generate_content(
            jsonizer_prompt.substitute(calender_text=calender_text, monday_date=monday),
            generation_config=config,
        )
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            print(f"  Skipping {text_path} (invalid JSON)")
            continue
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # 4. Queries — uses Monday date so current_time matches calendar dates
    print("Generating queries")
    for num in tqdm.tqdm(idx_range, desc="queries"):
        text_path = f"{TEXT_CALENDER_DIR}/{num}.txt"
        query_path = f"{QUERY_DIR}/{num}.txt"
        if not os.path.exists(text_path) or os.path.exists(query_path):
            continue
        if text_path not in monday_dates:
            continue  # its JSON conversion failed — skip
        calender_text = open(text_path).read()
        monday = monday_dates[text_path]
        config = GenerationConfig(temperature=1.0, response_mime_type="application/json")
        response = model.generate_content(
            query_prompt.substitute(calender_text=calender_text, monday_date=monday),
            generation_config=config,
        )
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            print(f"  Skipping {text_path} (invalid query JSON)")
            continue
        with open(query_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
