#!/usr/bin/env python3
"""Augment SFT trajectories via entity substitution and query paraphrasing.

Two independent augmentation methods:
1. Entity substitution (programmatic): replace emails, summaries, event IDs
2. Query paraphrasing (Gemini): rewrite user query + final response, keep tool chain

Usage:
    PYTHONPATH=src python scripts/data_generation/augment_trajectories.py
"""

import copy
import json
import glob
import os
import random
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerativeModel

from calendar_agent.paths import SFT_DATA_DIR as _SFT_DATA_DIR, CREDENTIALS_PATH

random.seed(42)

SFT_DATA_DIR = str(_SFT_DATA_DIR)
TRAJ_DIR = os.path.join(SFT_DATA_DIR, "trajectories")
OUT_DIR = os.path.join(SFT_DATA_DIR, "trajectories_augmented")
PROGRESS_FILE = os.path.join(OUT_DIR, "_progress.json")

MODEL_NAME = "gemini-2.0-flash-001"

# ── Category-weighted paraphrase counts ──────────────────────
CATEGORY_PARAPHRASE_COUNTS = {
    "Human Chaos (Edge Cases/Fragments)": 5,
    "Complex Logic & Conflict (Advanced)": 4,
    "Information Retrieval (Querying)": 2,
    "Schedule a Single Event": 3,
    "Vague & Contextual (Reasoning Required)": 3,
    "Modifier & Correction (Rescheduling/Updates)": 3,
    "Relative Time References (today, tomorrow, yesterday, this week)": 2,
}
DEFAULT_PARAPHRASE_COUNT = 3
N_ENTITY_VARIANTS = 2

# ── Name and domain pools for entity substitution ────────────
NAME_POOL = [
    "james.wilson", "sarah.chen", "michael.park", "emily.johnson", "david.kim",
    "jessica.liu", "robert.taylor", "amanda.garcia", "christopher.lee", "ashley.martinez",
    "daniel.wang", "stephanie.brown", "matthew.rodriguez", "lauren.nguyen", "andrew.thomas",
    "rachel.jackson", "joshua.white", "megan.harris", "ryan.clark", "nicole.lewis",
    "kevin.robinson", "jennifer.walker", "brian.hall", "kayla.allen", "eric.young",
    "heather.king", "jason.wright", "amber.lopez", "steven.hill", "tiffany.scott",
    "mark.green", "christina.adams", "anthony.baker", "melissa.nelson", "patrick.carter",
    "samantha.mitchell", "timothy.perez", "hannah.roberts", "john.turner", "kelly.phillips",
    "gregory.campbell", "victoria.parker", "joseph.evans", "natalie.edwards", "benjamin.collins",
    "elizabeth.stewart", "nicholas.sanchez", "vanessa.morris", "brandon.rogers", "danielle.reed",
    "tyler.cook", "allison.morgan", "justin.bailey", "katherine.rivera", "derek.cooper",
    "michelle.cox", "sean.howard", "olivia.ward", "aaron.torres", "courtney.peterson",
    "adam.gray", "cassandra.ramirez", "nathan.james", "brianna.watson", "ethan.brooks",
    "alexandra.kelly", "zachary.sanders", "brooke.price", "connor.bennett", "paige.wood",
    "marcus.hayes", "chelsea.ross", "dylan.henderson", "kimberly.coleman", "travis.jenkins",
    "lindsey.perry", "ian.powell", "morgan.long", "chad.patterson", "sierra.hughes",
]

DOMAIN_POOL = [
    "company.com", "corp.io", "enterprise.co", "firm.org", "team.dev",
    "global.net", "group.biz", "acme.com", "initech.co", "umbrella.org",
    "dynacorp.io", "nexus.dev", "vertex.co", "atlas.com", "pinnacle.net",
]

SUMMARY_POOLS = {
    "meeting": [
        "Team Standup", "Project Review", "Sprint Planning", "Architecture Review",
        "Design Sync", "Weekly Check-in", "Budget Review Meeting", "Strategy Session",
        "Quarterly Planning", "Product Demo", "Stakeholder Update", "Kickoff Meeting",
        "Retrospective", "Cross-Team Sync", "Executive Briefing", "Vendor Meeting",
        "Client Presentation", "Board Meeting", "Staff Meeting", "Department Huddle",
        "1-on-1 with Manager", "Performance Review", "Brainstorming Session",
        "Technical Discussion", "Release Planning", "Roadmap Review", "Sales Pipeline Review",
    ],
    "activity": [
        "Gym Session", "Yoga Class", "Team Lunch", "Coffee Chat", "Happy Hour",
        "Lunch Break", "Team Outing", "Walking Meeting", "Meditation Session",
        "Book Club", "Volunteer Event", "Networking Event", "Birthday Celebration",
        "Team Building", "Farewell Party", "Welcome Lunch", "Potluck",
    ],
    "appointment": [
        "Doctor Appointment", "Dentist Visit", "Car Service", "Haircut",
        "Parent Teacher Conference", "Home Inspection", "Tax Consultation",
        "Legal Consultation", "Insurance Review", "Financial Planning Session",
        "Interview", "Phone Screen", "Therapy Session", "Eye Exam",
    ],
    "work": [
        "Code Review", "Deploy Window", "System Maintenance", "Data Migration",
        "Training Session", "Onboarding", "Workshop", "Hackathon",
        "Documentation Sprint", "Bug Triage", "Incident Review", "Security Audit",
        "Focus Time", "Deep Work Block", "Research Time", "Writing Time",
    ],
}

ALL_SUMMARIES = []
for pool in SUMMARY_POOLS.values():
    ALL_SUMMARIES.extend(pool)


# ══════════════════════════════════════════════════════════════
# Entity Substitution
# ══════════════════════════════════════════════════════════════

def extract_entities(traj: dict) -> dict:
    """Extract emails, summaries, and event IDs from a trajectory."""
    traj_str = json.dumps(traj, default=str)

    emails = set(re.findall(r'[\w.+-]+@[\w.-]+\.\w+', traj_str))
    event_ids = set(re.findall(r'evt_[a-f0-9]+', traj_str))

    summaries = set()
    for day_events in (traj.get("calendar_before") or {}).values():
        for evt in day_events:
            if evt.get("summary"):
                summaries.add(evt["summary"])
    for day_events in (traj.get("calendar_after") or {}).values():
        for evt in day_events:
            if evt.get("summary"):
                summaries.add(evt["summary"])
    # Also from trajectory steps (tool call results)
    for step in traj.get("trajectory", []):
        if step.get("role") == "tool_call" and step.get("result"):
            # Skip get_current_time — its result is a timestamp, not a summary
            if step.get("name") == "get_current_time":
                continue
            result = step["result"]
            if isinstance(result, str):
                # New compact format: extract summaries from "id: evt_x | Summary — Day HH:MM-HH:MM" lines
                for m in re.finditer(r'id:\s*evt_[a-f0-9]+\s*\|\s*(.+?)\s*—', result):
                    summaries.add(m.group(1).strip())
                # Also extract from detail block first line (summary is the first line)
                lines = result.split('\n')
                if lines and not lines[0].startswith(('Found ', 'Event ', 'Error', 'RSVP ', 'id:')):
                    summaries.add(lines[0].strip())
            else:
                result_str = json.dumps(result, default=str)
                # Look for summary fields in result (legacy dict format)
                for m in re.finditer(r'"summary":\s*"([^"]+)"', result_str):
                    summaries.add(m.group(1))

    return {"emails": emails, "summaries": summaries, "event_ids": event_ids}


def classify_summary(summary: str) -> str:
    """Classify a summary into a pool category for replacement selection."""
    lower = summary.lower()
    meeting_words = ["meeting", "review", "sync", "standup", "planning", "demo",
                     "briefing", "update", "kickoff", "retrospective", "1-on-1",
                     "check-in", "huddle", "session", "discussion", "presentation"]
    activity_words = ["gym", "yoga", "lunch", "coffee", "happy hour", "outing",
                      "meditation", "book club", "volunteer", "networking",
                      "birthday", "celebration", "party", "potluck"]
    appointment_words = ["doctor", "dentist", "haircut", "appointment", "visit",
                         "consultation", "interview", "therapy", "exam", "inspection"]

    for w in activity_words:
        if w in lower:
            return "activity"
    for w in appointment_words:
        if w in lower:
            return "appointment"
    for w in meeting_words:
        if w in lower:
            return "meeting"
    return "work"


def build_substitution_map(entities: dict, rng: random.Random) -> dict:
    """Build a mapping of old entities -> new entities."""
    sub_map = {}

    # Substitute emails
    used_names = set()
    name_pool = list(NAME_POOL)
    rng.shuffle(name_pool)
    domain_pool = list(DOMAIN_POOL)
    rng.shuffle(domain_pool)

    for i, email in enumerate(sorted(entities["emails"])):
        if i < len(name_pool):
            name = name_pool[i]
        else:
            name = f"user{i}"
        domain = domain_pool[i % len(domain_pool)]
        sub_map[email] = f"{name}@{domain}"
        used_names.add(name)

    # Substitute summaries
    used_summaries = set()
    for summary in sorted(entities["summaries"]):
        cat = classify_summary(summary)
        pool = list(SUMMARY_POOLS.get(cat, SUMMARY_POOLS["work"]))
        rng.shuffle(pool)
        new_summary = None
        for candidate in pool:
            if candidate not in used_summaries and candidate != summary:
                new_summary = candidate
                break
        if new_summary is None:
            # Fall back to any pool
            all_shuffled = list(ALL_SUMMARIES)
            rng.shuffle(all_shuffled)
            for candidate in all_shuffled:
                if candidate not in used_summaries and candidate != summary:
                    new_summary = candidate
                    break
        if new_summary is None:
            new_summary = f"Event {rng.randint(1000, 9999)}"
        sub_map[summary] = new_summary
        used_summaries.add(new_summary)

    # Substitute event IDs
    for evt_id in sorted(entities["event_ids"]):
        new_hex = ''.join(rng.choices('0123456789abcdef', k=32))
        sub_map[evt_id] = f"evt_{new_hex}"

    return sub_map


def apply_substitution(traj: dict, sub_map: dict) -> dict:
    """Apply entity substitutions to the entire trajectory via string replacement."""
    traj_str = json.dumps(traj, default=str)

    # Sort replacements longest-first to avoid partial matches
    sorted_replacements = sorted(sub_map.items(), key=lambda x: -len(x[0]))

    for old, new in sorted_replacements:
        traj_str = traj_str.replace(old, new)

    new_traj = json.loads(traj_str)

    # Validate structure
    assert "trajectory" in new_traj, "Missing trajectory field after substitution"
    assert "query" in new_traj, "Missing query field after substitution"

    # Verify no old event IDs remain
    for old_key in sub_map:
        if old_key.startswith("evt_"):
            assert old_key not in json.dumps(new_traj, default=str), \
                f"Old event ID {old_key} still present after substitution"

    return new_traj


def generate_entity_variants(traj: dict, n: int, base_seed: int) -> list[dict]:
    """Generate n entity-substituted variants of a trajectory."""
    entities = extract_entities(traj)
    if not entities["emails"] and not entities["summaries"] and not entities["event_ids"]:
        return []

    variants = []
    for i in range(n):
        rng = random.Random(base_seed + i)
        sub_map = build_substitution_map(entities, rng)
        try:
            new_traj = apply_substitution(traj, sub_map)
            variants.append(new_traj)
        except (AssertionError, json.JSONDecodeError) as e:
            print(f"    Entity sub variant {i} failed: {e}")
    return variants


# ══════════════════════════════════════════════════════════════
# Query Paraphrasing (Gemini)
# ══════════════════════════════════════════════════════════════

def make_paraphrase_prompt(query: str, category: str, n: int) -> str:
    """Build the Gemini prompt for generating query paraphrases."""
    if "Human Chaos" in category:
        category_instruction = (
            "These are terse, fragmentary messages. Paraphrases must ALSO be terse — "
            "1-5 words, no complete sentences, typos OK. Examples of the style:\n"
            "  - 'move standup 2pm'\n"
            "  - 'cancel tmrw lunch'\n"
            "  - 'whats on wed'\n"
            "  - 'add meeting fri 3'"
        )
    else:
        category_instruction = "Vary between formal/casual, direct/polite, verbose/concise."

    return f"""You are rewriting calendar assistant queries. Generate {n} paraphrased versions.

Rules:
- Each paraphrase must request the EXACT same calendar action
- Same events, people, times, and days must be referenced
- Vary phrasing, sentence structure, politeness level, verbosity
- {category_instruction}
- Return a JSON array of {n} strings, nothing else

Original query: "{query}"
Category: {category}"""


def paraphrase_queries(model, query: str, category: str, n: int, max_retries: int = 3) -> list[str]:
    """Call Gemini to generate n paraphrases of a query."""
    prompt = make_paraphrase_prompt(query, category, n)

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()

            # Try to extract JSON array from response
            # Handle markdown code blocks
            if "```" in text:
                match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
                if match:
                    text = match.group(1).strip()

            paraphrases = json.loads(text)
            if isinstance(paraphrases, list) and len(paraphrases) >= 1:
                # Return up to n, all must be strings
                return [str(p) for p in paraphrases[:n]]

        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"    Paraphrase failed after {max_retries} attempts: {e}")

    return []


def regenerate_response(model, orig_query: str, orig_response: str, new_query: str,
                        max_retries: int = 2) -> str:
    """Ask Gemini to rephrase the assistant's final response for the new query."""
    prompt = f"""Rephrase this calendar assistant response to match a new query phrasing.

Rules:
- Keep ALL factual details identical: names, times, dates, event titles, confirmations
- Only change phrasing/structure to naturally respond to the new query
- Keep the same level of detail and length
- Do NOT add or remove any information
- Return ONLY the rephrased response, nothing else

Original query: "{orig_query}"
New query: "{new_query}"
Original response: "{orig_response}"

Rephrased response:"""

    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            if text:
                return text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            print(f"    Response regen failed: {e}")

    # Fallback: use original response
    return orig_response


def apply_paraphrase(traj: dict, new_query: str, new_response: str) -> dict:
    """Create a new trajectory with paraphrased query and response."""
    new_traj = copy.deepcopy(traj)
    new_traj["query"] = new_query

    # Update first step (user message)
    if new_traj["trajectory"] and new_traj["trajectory"][0]["role"] == "user":
        new_traj["trajectory"][0]["content"] = new_query

    # Update last assistant message
    for i in range(len(new_traj["trajectory"]) - 1, -1, -1):
        if new_traj["trajectory"][i]["role"] == "assistant":
            new_traj["trajectory"][i]["content"] = new_response
            break

    return new_traj


# ══════════════════════════════════════════════════════════════
# Progress tracking & IO
# ══════════════════════════════════════════════════════════════

def load_progress() -> dict:
    """Load progress file tracking completed augmentations."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": {}}


def save_progress(progress: dict):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def is_completed(progress: dict, cal_idx: str, traj_idx: int) -> bool:
    key = f"{cal_idx}:{traj_idx}"
    return key in progress.get("completed", {})


def mark_completed(progress: dict, cal_idx: str, traj_idx: int):
    key = f"{cal_idx}:{traj_idx}"
    progress.setdefault("completed", {})[key] = True


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Init Gemini
    creds_path = str(CREDENTIALS_PATH)
    with open(creds_path) as f:
        cd = json.load(f)
    creds = OAuth2Credentials(
        token=None,
        refresh_token=cd["refresh_token"],
        client_id=cd["client_id"],
        client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    vertexai.init(project="internal-ml-exp", location="us-central1", credentials=creds)
    model = GenerativeModel(MODEL_NAME)

    # Load all source trajectories
    all_source = []  # (cal_idx, traj_idx, traj_dict)
    for f_path in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.json"))):
        cal_idx = os.path.basename(f_path).replace(".json", "")
        trajs = json.load(open(f_path))
        for ti, traj in enumerate(trajs):
            all_source.append((cal_idx, ti, traj))

    print(f"Loaded {len(all_source)} source trajectories")

    # Category distribution
    from collections import Counter
    cat_counts = Counter(t[2].get("category", "unknown") for t in all_source)
    print("Source distribution:")
    for cat, cnt in cat_counts.most_common():
        n_para = CATEGORY_PARAPHRASE_COUNTS.get(cat, DEFAULT_PARAPHRASE_COUNT)
        expected = cnt * (1 + n_para + N_ENTITY_VARIANTS)
        print(f"  {cat}: {cnt} -> ~{expected}")
    print()

    progress = load_progress()

    # Organize output per calendar file
    output_data = {}  # cal_idx -> list of augmented trajectories

    # Load existing output if resuming
    for f_path in glob.glob(os.path.join(OUT_DIR, "*.json")):
        basename = os.path.basename(f_path)
        if basename.startswith("_"):
            continue
        cal_idx = basename.replace(".json", "")
        output_data[cal_idx] = json.load(open(f_path))

    total_generated = sum(len(v) for v in output_data.values())
    print(f"Existing augmented trajectories: {total_generated}")
    print()

    for source_idx, (cal_idx, traj_idx, traj) in enumerate(all_source):
        category = traj.get("category", "unknown")
        query = traj.get("query", "")
        n_paraphrases = CATEGORY_PARAPHRASE_COUNTS.get(category, DEFAULT_PARAPHRASE_COUNT)

        if is_completed(progress, cal_idx, traj_idx):
            continue

        print(f"[{source_idx+1}/{len(all_source)}] cal={cal_idx} traj={traj_idx} "
              f"cat=\"{category}\" para={n_paraphrases}")
        print(f"  query: {query[:80]}...")

        if cal_idx not in output_data:
            output_data[cal_idx] = []

        augmentation_base = {
            "source_file": f"{cal_idx}.json",
            "source_index": traj_idx,
        }

        # 1. Copy original
        orig = copy.deepcopy(traj)
        orig["augmentation"] = {**augmentation_base, "method": "original", "variant_index": 0}
        output_data[cal_idx].append(orig)

        # 2. Entity substitution variants
        base_seed = hash(f"{cal_idx}:{traj_idx}") & 0xFFFFFFFF
        entity_variants = generate_entity_variants(traj, N_ENTITY_VARIANTS, base_seed)
        for vi, variant in enumerate(entity_variants):
            variant["augmentation"] = {
                **augmentation_base, "method": "entity_substitution", "variant_index": vi
            }
            output_data[cal_idx].append(variant)
        print(f"  entity_sub: {len(entity_variants)} variants")

        # 3. Query paraphrasing
        paraphrases = paraphrase_queries(model, query, category, n_paraphrases)
        if paraphrases:
            # Find original final response
            orig_response = ""
            for step in reversed(traj.get("trajectory", [])):
                if step["role"] == "assistant":
                    orig_response = step["content"]
                    break

            for pi, new_query in enumerate(paraphrases):
                new_response = regenerate_response(model, query, orig_response, new_query)
                para_traj = apply_paraphrase(traj, new_query, new_response)
                para_traj["augmentation"] = {
                    **augmentation_base, "method": "paraphrase", "variant_index": pi
                }
                output_data[cal_idx].append(para_traj)
                time.sleep(0.3)  # Rate limiting

            print(f"  paraphrase: {len(paraphrases)} variants")
        else:
            print(f"  paraphrase: 0 variants (failed)")

        # Mark completed and save incrementally
        mark_completed(progress, cal_idx, traj_idx)
        save_progress(progress)

        # Save this calendar's output
        out_path = os.path.join(OUT_DIR, f"{cal_idx}.json")
        with open(out_path, "w") as f:
            json.dump(output_data[cal_idx], f, indent=2, default=str)

        time.sleep(0.3)

    # ── Final summary ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("AUGMENTATION SUMMARY")
    print("=" * 60)

    total = 0
    method_counts = Counter()
    cat_counts_aug = Counter()

    for cal_idx, trajs in sorted(output_data.items()):
        total += len(trajs)
        for t in trajs:
            aug = t.get("augmentation", {})
            method_counts[aug.get("method", "unknown")] += 1
            cat_counts_aug[t.get("category", "unknown")] += 1

    print(f"  Total augmented trajectories: {total}")
    print(f"  By method:")
    for method, cnt in method_counts.most_common():
        print(f"    {method}: {cnt}")
    print(f"  By category:")
    for cat, cnt in cat_counts_aug.most_common():
        print(f"    {cat}: {cnt}")
    print(f"  Output directory: {OUT_DIR}/")


if __name__ == "__main__":
    main()
