import warnings

warnings.filterwarnings("ignore")

import glob
import json
import os
import random
from datetime import date, timedelta

import tqdm
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerationConfig, GenerativeModel

from calendar_agent.prompts import architect_prompt, jsonizer_prompt, persona_prompt, query_prompt
from calendar_agent.paths import PROJECT_ROOT, CREDENTIALS_PATH


def random_monday(start_year: int = 2023, end_year: int = 2026) -> str:
    """Return a random Monday date string (YYYY-MM-DD) in the given year range."""
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    days_between = (end - start).days
    random_date = start + timedelta(days=random.randint(0, days_between))
    # Shift to the nearest Monday (weekday 0)
    monday = random_date - timedelta(days=random_date.weekday())
    return monday.strftime("%Y-%m-%d")


PERSONA_DIR = str(PROJECT_ROOT / "data" / "persona")
TEXT_CALENDER_DIR = str(PROJECT_ROOT / "data" / "calender")
JSON_CALENDER_DIR = str(PROJECT_ROOT / "data" / "json_calender")
QUERY_DIR = str(PROJECT_ROOT / "data" / "queries")

DATA_SIZE = 100
START_INDEX = 20  # Start numbering from 20 to avoid overwriting existing 0-19

if __name__ == "__main__":
    ########
    # Init #
    ########

    # Make sure all the dir's exist (don't wipe — we may be appending)
    for dir in (PERSONA_DIR, TEXT_CALENDER_DIR, JSON_CALENDER_DIR, QUERY_DIR):
        os.makedirs(dir, exist_ok=True)

    # Init genai model
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

    ###################
    # Data Generation #
    ###################

    # 50 professions for diverse calendar generation
    professions = [
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

    # 1. generate persona
    print(f"Generating {DATA_SIZE} personas (indices {START_INDEX}-{START_INDEX + DATA_SIZE - 1})")
    for num in tqdm.tqdm(range(START_INDEX, START_INDEX + DATA_SIZE)):
        out_path = f"{PERSONA_DIR}/{num}.txt"
        if os.path.exists(out_path):
            continue
        prof = random.choice(professions)
        response = model.generate_content(persona_prompt.substitute(profession=prof))
        open(out_path, "w").write(response.text)

    # 2. Generate text calender
    print("Generating text calenders")
    new_personas = [f"{PERSONA_DIR}/{num}.txt" for num in range(START_INDEX, START_INDEX + DATA_SIZE)]
    for file in tqdm.tqdm(new_personas):
        if not os.path.exists(file):
            continue
        out_path = file.replace(PERSONA_DIR, TEXT_CALENDER_DIR)
        if os.path.exists(out_path):
            continue

        # Extract persona
        persona = open(file, "r").read()
        if "#" not in persona:
            continue
        persona = persona.split("#")[-1]

        # Generate calender
        response = model.generate_content(architect_prompt.substitute(persona=persona))
        if "Monday" not in response.text:
            continue
        response = "Monday" + response.text.split("Monday")[1]
        # Verify all 7 days are present before writing
        required_days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        if not all(day in response for day in required_days):
            continue
        open(out_path, "w").write(response)

    # 3. Convert text calender to json
    # Save the Monday date per file so step 4 can use the same dates
    monday_dates = {}
    print("Converting text calenders to json")
    new_text_cals = [f"{TEXT_CALENDER_DIR}/{num}.txt" for num in range(START_INDEX, START_INDEX + DATA_SIZE)]
    for file in tqdm.tqdm(new_text_cals):
        if not os.path.exists(file):
            continue
        json_path = file.replace(TEXT_CALENDER_DIR, JSON_CALENDER_DIR)
        if os.path.exists(json_path):
            continue

        calender = open(file, "r").read()

        config = GenerationConfig(
            temperature=1.0, response_mime_type="application/json"
        )
        monday = random_monday()
        monday_dates[file] = monday
        response = model.generate_content(
            jsonizer_prompt.substitute(calender_text=calender, monday_date=monday),
            generation_config=config,
        )
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            print(f"  Skipping {file} (invalid JSON from model)")
            continue
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # 4. Generate queries
    # Pass the Monday date so Gemini generates current_time values that match
    # the actual calendar event dates from step 3.
    print("Generating queries")
    for file in tqdm.tqdm(new_text_cals):
        if not os.path.exists(file):
            continue
        if file not in monday_dates:
            continue  # JSON conversion failed for this calendar
        query_path = file.replace(TEXT_CALENDER_DIR, QUERY_DIR)
        if os.path.exists(query_path):
            continue

        calender = open(file, "r").read()
        monday = monday_dates[file]

        config = GenerationConfig(
            temperature=1.0, response_mime_type="application/json"
        )
        response = model.generate_content(
            query_prompt.substitute(calender_text=calender, monday_date=monday),
            generation_config=config,
        )
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            print(f"  Skipping {file} (invalid JSON from model)")
            continue
        with open(query_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
