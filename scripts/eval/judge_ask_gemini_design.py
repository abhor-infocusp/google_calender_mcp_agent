#!/usr/bin/env python3
"""Meta-prompt: ask Gemini what information IT would want to judge IR queries.

Shows Gemini a representative spread of Information-Retrieval cases (mix of
easy + the 3 cases it currently fails on) and asks an open-ended design
question: what additional structured fields would you want surfaced in the
judging prompt to make consistent verdicts?

This is a design-elicitation step — we don't ask Gemini for verdicts here,
we ask it what it needs.
"""
from __future__ import annotations
import json
from pathlib import Path

import google.auth.transport.requests
import vertexai
from google.oauth2.credentials import Credentials as OAuth2Credentials
from vertexai.generative_models import GenerationConfig, GenerativeModel

import sys
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from calendar_agent.paths import CREDENTIALS_PATH
from calendar_agent.judge.features import extract_features
from calendar_agent.judge.structured_prompts import build_ir

PROJECT = "internal-ml-exp"
LOCATION = "us-central1"
MODEL = "gemini-2.0-flash-001"
GEN_CFG = GenerationConfig(temperature=0.0, top_p=1.0, max_output_tokens=4096)

INPUT_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_review_input.jsonl"
TRUTH_JSONL = REPO / "runs/judge_baseline_20260430/eval/manual_verdicts_relabeled.jsonl"


def init_vertex():
    cd = json.load(open(CREDENTIALS_PATH))
    creds = OAuth2Credentials(
        token=None,
        refresh_token=cd["refresh_token"],
        client_id=cd["client_id"],
        client_secret=cd["client_secret"],
        token_uri="https://oauth2.googleapis.com/token",
    )
    creds.refresh(google.auth.transport.requests.Request())
    vertexai.init(project=PROJECT, location=LOCATION, credentials=creds)


# Selection of cases to show Gemini:
# 3 currently FAILING cases (Gemini got these wrong with the current structured prompt)
# 3 currently CORRECT cases for contrast
SAMPLE_SIDS = {
    # Failures
    "cal_32_q_7": "FAIL — Gemini said Correct, gt=Incorrect (agent claimed 'attendees not listed' when they ARE in calendar)",
    "cal_22_q_7": "FAIL — Gemini said Incorrect, gt=Correct (agent named 'Lunch with Mentor' — title implies attendee)",
    "cal_20_q_7": "FAIL — Gemini said Correct, gt=Incorrect (agent gave vague 'with friends or family' instead of attendee names)",
    # Successes (Gemini got these right with current prompt)
    "cal_18_q_6": "OK — easy presence-check, gt=Incorrect, judge=Incorrect",
    "cal_19_q_6": "OK — attendee query, gt=Correct, judge=Correct",
    "cal_16_q_6": "OK — time query, gt=Incorrect, judge=Incorrect",
}


def case_block(rec, label):
    feats = extract_features(rec)
    sys_p, user_p, _ = build_ir(rec)
    return f"""[CASE — {label}]
sid: {rec['sid']}

PROMPT THE JUDGE CURRENTLY SEES:
{user_p}

GROUND TRUTH (canonical manual label): {rec['_gt']}
"""


META_PROMPT = """\
You are designing a judging system for a calendar assistant. The judge has
to decide whether the agent's response to an Information-Retrieval (IR)
query — "what time is X?", "who is invited to Y?", "do I have anything
on Tuesday?" — is correct.

Below are 6 real IR cases. The judge currently sees the prompt shown for
each. Some of these the judge gets right; some it gets wrong. The ground
truth at the bottom of each case is the canonical manual label.

Your task is NOT to grade these cases. Your task is to TELL ME what
*additional structured fields* you would want pre-computed and surfaced
in the prompt so that you could judge IR queries reliably across the
board. Be specific about:

  1. What new fields to add (give exact names + what they contain).
  2. Why each field would help — name a failure mode it prevents.
  3. Which existing fields could be DROPPED if redundant.
  4. Any rules to add to the system prompt.

Aim for the smallest, sharpest set of fields. Each field must earn its
place by fixing a specific mistake.

After your design, write a CONCRETE, COMPLETE rewritten IR system prompt
+ user-prompt template that you would use going forward.

────────────────────────────────────────────────────────────
"""


def main():
    init_vertex()
    inputs = [json.loads(l) for l in INPUT_JSONL.open()]
    truth = [json.loads(l) for l in TRUTH_JSONL.open()]
    by_sid = {r["sid"]: (i, r) for i, r in enumerate(inputs)}

    blocks = [META_PROMPT]
    for sid, label in SAMPLE_SIDS.items():
        if sid not in by_sid:
            print(f"[skip] {sid} not in inputs")
            continue
        i, rec = by_sid[sid]
        rec["_gt"] = truth[i]["verdict"]
        blocks.append(case_block(rec, label))
        blocks.append("\n────────────────────────────────────────────────────────────\n")

    blocks.append("""
Now produce your design. Format as:

# DIAGNOSIS
(2-4 sentences on why the judge fails on these cases.)

# NEW FIELDS
(Bullet list: field_name — what it contains — failure mode it prevents.)

# DROPPED FIELDS
(If any.)

# REWRITTEN SYSTEM PROMPT

# REWRITTEN USER PROMPT TEMPLATE
""")

    full_prompt = "".join(blocks)
    print(f"[prompt size: {len(full_prompt)} chars]")
    out_path = REPO / "runs/judge_meta_design_ir_20260507"
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "request.txt").write_text(full_prompt)

    model = GenerativeModel(MODEL)
    print("[asking Gemini...]")
    resp = model.generate_content(full_prompt, generation_config=GEN_CFG)
    answer = resp.text
    (out_path / "response.md").write_text(answer)
    print(f"\n[saved to {out_path}]")
    print("=" * 70)
    print(answer)


if __name__ == "__main__":
    main()
