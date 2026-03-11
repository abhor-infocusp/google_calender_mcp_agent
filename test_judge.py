"""Quick test for the eval judge prompt."""
import vertexai, json, sys
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerativeModel
from run_trajectory import EVAL_SYSTEM_PROMPT

with open("gcloud_credentials.json") as f:
    cd = json.load(f)
creds = OAuth2Credentials(
    token=None, refresh_token=cd["refresh_token"],
    client_id=cd["client_id"], client_secret=cd["client_secret"],
    token_uri="https://oauth2.googleapis.com/token",
)
vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
em = GenerativeModel("gemini-2.0-flash-001", system_instruction=[EVAL_SYSTEM_PROMPT])


def judge(query, response, expected, before, after):
    bt = before
    at = after
    prompt = (
        f"Query: {query}\n\nResponse: {response}\n\nExpected: {expected}\n\n"
        f"Before:\n{bt}\n\nAfter:\n{at}\n\n"
        "Was the task completed correctly? End with one word: Correct or Incorrect."
    )
    try:
        r = em.generate_content(prompt)
        text = r.text.strip()
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for line in reversed(lines):
            ll = line.lower()
            if ll == "incorrect": return "Incorrect"
            if ll == "correct": return "Correct"
        for line in reversed(lines):
            ll = line.lower()
            if "incorrect" in ll: return "Incorrect"
            if "correct" in ll: return "Correct"
        return "Incorrect"
    except Exception as e:
        return f"ERROR: {e}"


tests = [
    ("Correct action (event created)",
     "Add a meeting with John on Wednesday at 2 PM.",
     "Done! Created 'Meeting with John' on Wednesday 2:00 PM.",
     "Creates a new event named 'Meeting with John' on Wednesday at 2 PM.",
     "Wednesday:\n  09:00-17:00  Work",
     "Wednesday:\n  09:00-17:00  Work\n  14:00-15:00  Meeting with John",
     "Correct"),

    ("Asked clarification instead of acting",
     "Add a meeting with John on Wednesday at 2 PM.",
     "What is John's email? How long should the meeting be?",
     "Creates a new event named 'Meeting with John' on Wednesday at 2 PM.",
     "Wednesday:\n  09:00-17:00  Work",
     "Wednesday:\n  09:00-17:00  Work",
     "Incorrect"),

    ("Correct info retrieval",
     "What am I doing on Tuesday?",
     "On Tuesday: Lifeguard Duty 9AM-5PM, Marine Biology Course 6:30-7:30PM.",
     "Lists all Tuesday events.",
     "Tuesday:\n  09:00-17:00  Lifeguard Duty\n  18:30-19:30  Marine Biology Course",
     "Tuesday:\n  09:00-17:00  Lifeguard Duty\n  18:30-19:30  Marine Biology Course",
     "Correct"),

    ("Wrong info returned",
     "What am I doing on Tuesday?",
     "You have no events on Tuesday.",
     "Lists all Tuesday events.",
     "Tuesday:\n  09:00-17:00  Lifeguard Duty\n  18:30-19:30  Marine Biology Course",
     "Tuesday:\n  09:00-17:00  Lifeguard Duty\n  18:30-19:30  Marine Biology Course",
     "Incorrect"),

    ("Correct deletion",
     "Cancel my swim practice on Wednesday.",
     "Done, cancelled Morning Swim on Wednesday.",
     "Deletes the 'Morning Swim' event on Wednesday.",
     "Wednesday:\n  07:30-08:00  Morning Swim\n  09:00-17:00  Lifeguard Duty",
     "Wednesday:\n  09:00-17:00  Lifeguard Duty",
     "Correct"),

    ("Partial completion (wrong time)",
     "Add a meeting with John on Wednesday at 2 PM.",
     "Created 'Meeting with John' on Wednesday at 10 AM.",
     "Creates a new event named 'Meeting with John' on Wednesday at 2 PM.",
     "Wednesday:\n  09:00-17:00  Work",
     "Wednesday:\n  09:00-17:00  Work\n  10:00-11:00  Meeting with John",
     "Incorrect"),
]

passed = 0
for i, (name, q, r, e, b, a, want) in enumerate(tests, 1):
    v = judge(q, r, e, b, a)
    ok = v == want
    if ok:
        passed += 1
    status = "PASS" if ok else "FAIL"
    print(f"Test {i}: {status} | got={v} want={want} | {name}", flush=True)

print(f"\n{passed}/{len(tests)} passed")
