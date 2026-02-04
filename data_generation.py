import warnings
warnings.filterwarnings("ignore")

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from prompt import persona_prompt, architect_prompt, jsonizer_prompt, query_prompt
import tqdm
import random
import glob, os
import shutil
import json

PERSONA_DIR = "data/persona"
TEXT_CALENDER_DIR = "data/calender"
JSON_CALENDER_DIR = "data/json_calender"
QUERY_DIR = "data/queries"

DATA_SIZE = 2

if __name__ == "__main__":
    ########
    # Init #
    ########

    # Make sure all the dir's exist and are empty
    for dir in (PERSONA_DIR, TEXT_CALENDER_DIR, JSON_CALENDER_DIR, QUERY_DIR):
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)


    # Init genai model
    vertexai.init(project="internal-ml-exp", location="us-central1")
    model = GenerativeModel("gemini-2.0-flash-001", system_instruction=[
            "You are a helpful, creative assistant.",
            "Avoid repetitive sentence structures.",
            "Be concise but maintain a natural, conversational flow."
        ])


    ###################
    # Data Generation #
    ###################

    # A list of professions to start from
    # Without this in the code, the model tends to produce very similar professions on each personal prompt
    professions = "Teacher, Accountant, Nurse, Driver, Engineer, Manager, Clerk, Salesperson, Analyst, Technician, Plumber, Electrician, Chef, Artist, Writer, Lawyer, Doctor, Pilot, Farmer, Guard".split(", ")

    # 1. generate persona
    print("Generating personas")
    for num in tqdm.tqdm(range(DATA_SIZE)):
        prof = professions[random.randint(0, len(professions)-1)]
        response = model.generate_content(persona_prompt.substitute(profession=prof))
        open(f"{PERSONA_DIR}/{num}.txt", 'w').write(response.text)

    # 2. Generate text calender
    print("Generating text calenders")
    for file in tqdm.tqdm(glob.glob(f"{PERSONA_DIR}/*")):
        # Extract persona
        persona = open(file, 'r').read()
        if "#" not in persona:
            continue
        persona = persona.split("#")[-1]

        # Generate calender
        response = model.generate_content(architect_prompt.substitute(persona=persona))
        if "Monday" not in response.text:
            continue
        response = "Monday" + response.text.split("Monday")[1]
        open(file.replace(PERSONA_DIR, TEXT_CALENDER_DIR), 'w').write(response)

    # 3. Convert text calender to json
    print("Converting text calenders to json")
    for file in tqdm.tqdm(glob.glob(f"{TEXT_CALENDER_DIR}/*")):
        calender = open(file, 'r').read()

        # Generate calender
        config = GenerationConfig(
            temperature=1.0, # Or your preferred temp
            response_mime_type="application/json"
        )
        response = model.generate_content(jsonizer_prompt.substitute(calender_text=calender), generation_config=config)
        data = json.loads(response.text)
        with open(file.replace(TEXT_CALENDER_DIR, JSON_CALENDER_DIR), 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # 4. Generate queries
    print("Converting text calenders to json")
    for file in tqdm.tqdm(glob.glob(f"{TEXT_CALENDER_DIR}/*")):
        calender = open(file, 'r').read()

        # Generate calender
        config = GenerationConfig(
            temperature=1.0, # Or your preferred temp
            response_mime_type="application/json"
        )
        response = model.generate_content(query_prompt.substitute(calender_text=calender), generation_config=config)
        data = json.loads(response.text)
        with open(file.replace(TEXT_CALENDER_DIR, QUERY_DIR), 'w', encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

