"""Count DPO / KTO / RFT training signal available in stored ART rollouts.

For each stored step (one scenario × 8 rollouts), classify:
  mixed     : ≥1 correct and ≥1 wrong  → DPO pair + KTO both labels + RFT correct(s)
  all_correct: all 8 correct            → RFT only (no preference signal)
  all_wrong : all 8 wrong               → KTO negatives only
  degenerate: empty / error             → unusable

Also report:
  - Per-category breakdown
  - DPO pair count (1-per-step vs N_corr × N_wrong max)
  - RFT trajectory count
  - KTO label balance
"""

import glob
import json
import os
from collections import Counter, defaultdict

import pandas as pd

from calendar_agent.paths import PROJECT_ROOT

TRAJ_DIR = str(PROJECT_ROOT / ".art" / "calendar-agent" / "models" / "calendar-agent-001" / "trajectories" / "train")

CATEGORY_SHORT = {
    "Complex Logic & Conflict (Advanced)": "Complex",
    "Human Chaos (Edge Cases/Fragments)": "Chaos",
    "Information Retrieval (Querying)": "IR",
    "Modifier & Correction (Rescheduling/Updates)": "Modifier",
    "Relative Time References (today, tomorrow, yesterday, this week)": "RelTime",
    "Schedule a Single Event": "Schedule",
    "Vague & Contextual (Reasoning Required)": "Vague",
}


def main():
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.parquet")))
    print(f"Scanning {len(files)} parquet files under {TRAJ_DIR}")

    total_steps = 0
    mixed = 0
    all_correct = 0
    all_wrong = 0
    degenerate = 0

    dpo_pairs_1per = 0        # 1 pair per mixed step
    dpo_pairs_max = 0          # N_corr × N_wrong per mixed step
    rft_correct_trajs = 0      # total correct rollouts (usable for RFT/SFT)
    kto_positive = 0           # total correct rollouts
    kto_negative = 0           # total wrong rollouts

    per_cat: dict[str, dict] = defaultdict(lambda: {
        "steps": 0, "mixed": 0, "all_correct": 0, "all_wrong": 0,
        "correct_trajs": 0, "wrong_trajs": 0,
        "dpo_pairs_1per": 0, "dpo_pairs_max": 0,
    })
    # Track repeat coverage of scenarios
    scenarios_seen: Counter = Counter()
    scenarios_with_mixed: Counter = Counter()

    for f in files:
        try:
            df = pd.read_parquet(f, columns=["reward", "metadata"])
        except Exception as e:
            print(f"  skip {f}: {e}")
            continue
        if len(df) == 0:
            degenerate += 1
            continue
        # All rows in a file share the same scenario (by construction)
        meta0 = df.iloc[0]["metadata"]
        if isinstance(meta0, str):
            meta0 = json.loads(meta0)
        scenario_id = meta0.get("scenario_id", "?")
        cat_full = meta0.get("category", "Unknown")
        cat = CATEGORY_SHORT.get(cat_full, "Unk")

        rewards = df["reward"].tolist()
        n_corr = sum(1 for r in rewards if r == 1.0)
        n_wrong = sum(1 for r in rewards if r == 0.0)
        if n_corr + n_wrong == 0:
            degenerate += 1
            continue

        total_steps += 1
        scenarios_seen[scenario_id] += 1
        pc = per_cat[cat]
        pc["steps"] += 1
        pc["correct_trajs"] += n_corr
        pc["wrong_trajs"] += n_wrong
        rft_correct_trajs += n_corr
        kto_positive += n_corr
        kto_negative += n_wrong

        if n_corr > 0 and n_wrong > 0:
            mixed += 1
            pc["mixed"] += 1
            dpo_pairs_1per += 1
            pair_max_here = n_corr * n_wrong
            dpo_pairs_max += pair_max_here
            pc["dpo_pairs_1per"] += 1
            pc["dpo_pairs_max"] += pair_max_here
            scenarios_with_mixed[scenario_id] += 1
        elif n_corr > 0:
            all_correct += 1
            pc["all_correct"] += 1
        else:
            all_wrong += 1
            pc["all_wrong"] += 1

    # ── Report ──
    print()
    print(f"{'='*60}")
    print(f"DPO / KTO / RFT signal across {total_steps} stored rollout-groups")
    print(f"{'='*60}")
    print(f"  Mixed (DPO-usable)      : {mixed:>6d}  ({mixed/total_steps*100:5.1f}%)")
    print(f"  All correct (RFT only)  : {all_correct:>6d}  ({all_correct/total_steps*100:5.1f}%)")
    print(f"  All wrong (KTO neg only): {all_wrong:>6d}  ({all_wrong/total_steps*100:5.1f}%)")
    print(f"  Degenerate              : {degenerate:>6d}")
    print()
    print(f"  Unique scenarios seen   : {len(scenarios_seen):>6d}")
    print(f"  Scenarios ever mixed    : {len(scenarios_with_mixed):>6d}")
    print()
    print(f"  DPO pairs (1 / step)    : {dpo_pairs_1per:>6d}")
    print(f"  DPO pairs (N_c × N_w)   : {dpo_pairs_max:>6d}")
    print(f"  RFT correct trajectories: {rft_correct_trajs:>6d}")
    print(f"  KTO positives           : {kto_positive:>6d}")
    print(f"  KTO negatives           : {kto_negative:>6d}  (balance={kto_positive/(kto_positive+kto_negative):.2%} pos)")
    print()
    print(f"{'Category':<10s} {'Steps':>6s} {'Mixed':>6s} {'AllC':>5s} {'AllW':>5s} {'Corr':>6s} {'Wrong':>6s} {'DPO(1)':>7s} {'DPO(max)':>9s}")
    print("-" * 70)
    for cat in sorted(per_cat.keys()):
        d = per_cat[cat]
        print(f"{cat:<10s} {d['steps']:>6d} {d['mixed']:>6d} {d['all_correct']:>5d} {d['all_wrong']:>5d} "
              f"{d['correct_trajs']:>6d} {d['wrong_trajs']:>6d} {d['dpo_pairs_1per']:>7d} {d['dpo_pairs_max']:>9d}")

    # Save machine-readable summary
    out_path = PROJECT_ROOT / "runs" / "analysis" / "dpo_pair_counts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "trajectory_dir": TRAJ_DIR,
        "total_steps": total_steps,
        "mixed": mixed,
        "all_correct": all_correct,
        "all_wrong": all_wrong,
        "degenerate": degenerate,
        "unique_scenarios": len(scenarios_seen),
        "scenarios_with_mixed": len(scenarios_with_mixed),
        "dpo_pairs_1per_step": dpo_pairs_1per,
        "dpo_pairs_max": dpo_pairs_max,
        "rft_correct_trajectories": rft_correct_trajs,
        "kto_positives": kto_positive,
        "kto_negatives": kto_negative,
        "per_category": dict(per_cat),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
