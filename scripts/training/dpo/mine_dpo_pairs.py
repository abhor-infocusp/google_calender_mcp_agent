"""Mine DPO training pairs from stored ART rollout parquets.

For each parquet (= one scenario, 8 rollouts) with at least 1 correct and
1 incorrect rollout, emit one pair (random correct as `chosen`, random
incorrect as `rejected`) in TRL conversational DPO format.

Output schema (one JSON per line):
    {
      "prompt":   [ {role, content}, ... ],
      "chosen":   [ {role, content [, tool_calls, tool_call_id]}, ... ],
      "rejected": [ {role, content [, tool_calls, tool_call_id]}, ... ],
      "metadata": { scenario_id, step, category, complexity }
    }

Shared prefix (system + user) goes in `prompt`; everything after goes in
chosen/rejected. `tool_calls` stored in parquets as a JSON string is
parsed back into native list form so the tokenizer's chat template can
consume it.
"""

import argparse
import glob
import json
import os
import random
import sys
from collections import defaultdict

import pandas as pd

from calendar_agent.paths import PROJECT_ROOT

TRAJ_DIR = str(PROJECT_ROOT / ".art" / "calendar-agent" / "models" / "calendar-agent-001" / "trajectories" / "train")


def normalize_message(m: dict) -> dict:
    """Convert one stored message into TRL-compatible format.

    Parses `tool_calls` from JSON string → list, drops None fields,
    drops the `trainable` bookkeeping flag (not used by chat template).
    """
    out = {"role": m["role"], "content": m.get("content") or ""}
    tc = m.get("tool_calls")
    if tc:
        if isinstance(tc, str):
            try:
                tc_list = json.loads(tc)
            except json.JSONDecodeError:
                tc_list = None
        else:
            tc_list = tc
        if tc_list:
            out["tool_calls"] = tc_list
    tc_id = m.get("tool_call_id")
    if tc_id:
        out["tool_call_id"] = tc_id
    return out


def split_prompt_and_completion(messages: list) -> tuple[list, list]:
    """Return (prompt, completion) where prompt = leading system+user messages."""
    prompt_end = 0
    for i, m in enumerate(messages):
        if m["role"] in ("system", "user"):
            prompt_end = i + 1
        else:
            break
    return messages[:prompt_end], messages[prompt_end:]


def mine_pairs(
    parquet_dir: str,
    output_path: str,
    seed: int = 42,
    max_pairs_per_scenario: int = 1,
    min_completion_turns: int = 1,
) -> dict:
    rng = random.Random(seed)
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"Scanning {len(files)} parquets in {parquet_dir}")

    per_cat: dict[str, int] = defaultdict(int)
    total_pairs = 0
    total_scanned = 0
    total_mixed = 0
    skipped_malformed = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as out:
        for f in files:
            try:
                df = pd.read_parquet(f, columns=["reward", "messages", "metadata"])
            except Exception as e:
                skipped_malformed += 1
                continue
            if len(df) == 0:
                continue
            total_scanned += 1

            meta0 = df.iloc[0]["metadata"]
            if isinstance(meta0, str):
                meta0 = json.loads(meta0)

            correct_rows = df[df["reward"] == 1.0]
            wrong_rows = df[df["reward"] == 0.0]

            if len(correct_rows) == 0 or len(wrong_rows) == 0:
                continue
            total_mixed += 1

            # Pick up to `max_pairs_per_scenario` pairs per mixed scenario.
            # With 1: random.choice each side. With >1: cartesian-sample.
            for _ in range(max_pairs_per_scenario):
                c_idx = rng.choice(correct_rows.index.tolist())
                w_idx = rng.choice(wrong_rows.index.tolist())
                c_msgs = [normalize_message(dict(m)) for m in df.loc[c_idx]["messages"]]
                w_msgs = [normalize_message(dict(m)) for m in df.loc[w_idx]["messages"]]

                c_prompt, c_completion = split_prompt_and_completion(c_msgs)
                w_prompt, w_completion = split_prompt_and_completion(w_msgs)

                if not c_completion or not w_completion:
                    skipped_malformed += 1
                    continue
                if c_prompt != w_prompt:
                    # Should be identical since same scenario — but guard anyway.
                    # Pick one (the correct one's prompt) and warn.
                    pass

                if (
                    sum(1 for m in c_completion if m["role"] == "assistant") < min_completion_turns
                    or sum(1 for m in w_completion if m["role"] == "assistant") < min_completion_turns
                ):
                    skipped_malformed += 1
                    continue

                record = {
                    "prompt": c_prompt,
                    "chosen": c_completion,
                    "rejected": w_completion,
                    "metadata": {
                        "scenario_id": meta0.get("scenario_id"),
                        "step": meta0.get("step"),
                        "category": meta0.get("category"),
                        "complexity": meta0.get("complexity"),
                        "parquet": os.path.basename(f),
                    },
                }
                out.write(json.dumps(record) + "\n")
                total_pairs += 1
                per_cat[meta0.get("category", "Unknown")] += 1

    summary = {
        "parquets_scanned": total_scanned,
        "mixed_scenarios": total_mixed,
        "pairs_written": total_pairs,
        "skipped_malformed": skipped_malformed,
        "per_category": dict(per_cat),
        "output": output_path,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", default=TRAJ_DIR)
    ap.add_argument("--output", default=str(PROJECT_ROOT / "runs" / "dpo" / "pairs_from_14b_rl.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pairs-per-scenario", type=int, default=1)
    args = ap.parse_args()

    mine_pairs(
        parquet_dir=args.parquet_dir,
        output_path=args.output,
        seed=args.seed,
        max_pairs_per_scenario=args.max_pairs_per_scenario,
    )


if __name__ == "__main__":
    main()
