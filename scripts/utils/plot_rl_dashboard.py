#!/usr/bin/env python3
"""Plot RL training dashboard from ART trajectory files.

Panels:
  1. Reward curve with MA-10
  2. Skip breakdown (all-correct vs all-wrong vs trained)
  3. Category-wise reward trends (MA-10 windows of ~80 trajectories)
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from calendar_agent.paths import PROJECT_ROOT

TRAJ_DIR = str(
    PROJECT_ROOT
    / ".art"
    / "calendar-agent"
    / "models"
    / "calendar-agent-001"
    / "trajectories"
    / "train"
)
WINDOW = 10


def load_trajectory_data():
    """Load per-step data from ART trajectory JSONL files."""
    pattern = os.path.join(TRAJ_DIR, "*.jsonl")
    steps = []

    for f in sorted(glob.glob(pattern)):
        step = int(os.path.basename(f).replace(".jsonl", ""))
        trajectories = []
        with open(f) as fh:
            for line in fh:
                data = json.loads(line)
                for t in data.get("trajectories", []):
                    trajectories.append(
                        {
                            "reward": t.get("reward", 0.0),
                            "category": t.get("metadata", {}).get("category", "Unknown"),
                            "complexity": t.get("metadata", {}).get("complexity", "Unknown"),
                            "had_error": t.get("metrics", {}).get("had_error", 0),
                            "no_final_answer": t.get("metrics", {}).get("no_final_answer", 0),
                            "context_overflow": t.get("metrics", {}).get("context_overflow", 0),
                        }
                    )

        rewards = [t["reward"] for t in trajectories]
        avg_reward = np.mean(rewards) if rewards else 0.0
        all_same = len(set(rewards)) <= 1
        all_correct = all(r == 1.0 for r in rewards)
        all_wrong = all(r == 0.0 for r in rewards)
        has_overflow = any(t["context_overflow"] for t in trajectories)

        steps.append(
            {
                "step": step,
                "avg_reward": avg_reward,
                "n_trajectories": len(trajectories),
                "trajectories": trajectories,
                "skipped": all_same,
                "skip_type": (
                    "all_correct"
                    if all_correct and all_same
                    else "all_wrong"
                    if all_wrong and all_same
                    else "trained"
                ),
                "has_overflow": has_overflow,
            }
        )

    steps.sort(key=lambda s: s["step"])
    return steps


def shorten_category(cat):
    """Shorten long category names for legend."""
    # Take first 30 chars, or abbreviate common patterns
    if len(cat) <= 35:
        return cat
    # Try to find a parenthetical and remove it
    if "(" in cat:
        cat = cat[: cat.index("(")].strip()
    if len(cat) <= 35:
        return cat
    return cat[:32] + "..."


def main():
    steps = load_trajectory_data()
    if not steps:
        print("No trajectory data found.")
        return

    print(f"Loaded {len(steps)} steps with {sum(s['n_trajectories'] for s in steps)} total trajectories")

    step_nums = [s["step"] for s in steps]
    rewards = [s["avg_reward"] for s in steps]

    # ── Panel 1: Reward curve with MA ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 14))

    ax1 = axes[0]
    ax1.scatter(step_nums, rewards, alpha=0.2, s=12, color="tab:blue", label="Per step", zorder=2)

    if len(rewards) >= WINDOW:
        kernel = np.ones(WINDOW) / WINDOW
        ma = np.convolve(rewards, kernel, mode="valid")
        ma_steps = step_nums[WINDOW - 1 :]
        ax1.plot(ma_steps, ma, color="tab:blue", linewidth=2.5, label=f"MA-{WINDOW}", zorder=3)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Avg Reward")
    ax1.set_title(f"Reward Curve (MA-{WINDOW})")
    ax1.legend(loc="upper left")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Skip breakdown (stacked area) ──
    ax2 = axes[1]

    # Compute rolling counts in windows
    skip_steps = []
    trained_pct = []
    all_correct_pct = []
    all_wrong_pct = []

    for i in range(len(steps)):
        window_start = max(0, i - WINDOW + 1)
        window = steps[window_start : i + 1]
        n = len(window)
        n_trained = sum(1 for s in window if s["skip_type"] == "trained")
        n_correct = sum(1 for s in window if s["skip_type"] == "all_correct")
        n_wrong = sum(1 for s in window if s["skip_type"] == "all_wrong")
        skip_steps.append(steps[i]["step"])
        trained_pct.append(n_trained / n * 100)
        all_correct_pct.append(n_correct / n * 100)
        all_wrong_pct.append(n_wrong / n * 100)

    ax2.stackplot(
        skip_steps,
        trained_pct,
        all_correct_pct,
        all_wrong_pct,
        labels=["Trained", "Skip (all correct)", "Skip (all wrong)"],
        colors=["tab:green", "tab:blue", "tab:red"],
        alpha=0.7,
    )
    ax2.set_xlabel("Step")
    ax2.set_ylabel("% of steps (rolling window)")
    ax2.set_title(f"Step Outcome Breakdown (rolling {WINDOW}-step window)")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Category-wise reward trends ──
    ax3 = axes[2]

    # Collect all trajectories with step info
    all_trajs = []
    for s in steps:
        for t in s["trajectories"]:
            all_trajs.append({"step": s["step"], **t})

    # Group by category
    cat_trajs = defaultdict(list)
    for t in all_trajs:
        cat_trajs[t["category"]].append(t)

    # For each category, compute rolling average in windows of steps
    # Use the same WINDOW of steps (each with ~8 trajectories = ~80 per window)
    cat_colors = {}
    cmap = plt.cm.get_cmap("tab20", len(cat_trajs))

    # Sort categories by total count (most common first)
    sorted_cats = sorted(cat_trajs.keys(), key=lambda c: -len(cat_trajs[c]))

    # Only plot categories with >= 10 trajectories
    plotted = 0
    for idx, cat in enumerate(sorted_cats):
        trajs = sorted(cat_trajs[cat], key=lambda t: t["step"])
        if len(trajs) < 10:
            continue

        # Compute MA over step-windows
        # Group by step first
        step_rewards = defaultdict(list)
        for t in trajs:
            step_rewards[t["step"]].append(t["reward"])

        cat_steps = sorted(step_rewards.keys())
        cat_avg = [np.mean(step_rewards[s]) for s in cat_steps]

        if len(cat_avg) >= WINDOW:
            kernel = np.ones(WINDOW) / WINDOW
            ma = np.convolve(cat_avg, kernel, mode="valid")
            ma_x = cat_steps[WINDOW - 1 :]
            label = f"{shorten_category(cat)} ({len(trajs)})"
            ax3.plot(ma_x, ma, linewidth=1.8, label=label, color=cmap(idx), alpha=0.85)
            plotted += 1
        elif len(cat_avg) >= 3:
            # Too few for MA, plot raw
            label = f"{shorten_category(cat)} ({len(trajs)})"
            ax3.plot(cat_steps, cat_avg, linewidth=1, label=label, color=cmap(idx), alpha=0.5, linestyle="--")
            plotted += 1

    ax3.set_xlabel("Step")
    ax3.set_ylabel("Avg Reward")
    ax3.set_title(f"Category-wise Reward (MA-{WINDOW}, n=trajectories)")
    if plotted <= 15:
        ax3.legend(loc="upper left", fontsize=7, ncol=2)
    else:
        ax3.legend(loc="upper left", fontsize=6, ncol=3)
    ax3.set_ylim(-0.05, 1.05)
    ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = str(PROJECT_ROOT / "rl_dashboard.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # Print category summary
    print(f"\n{'Category':<60s} {'Count':>5s} {'Avg Reward':>10s}")
    print("-" * 80)
    for cat in sorted_cats:
        trajs = cat_trajs[cat]
        avg = np.mean([t["reward"] for t in trajs])
        print(f"{shorten_category(cat):<60s} {len(trajs):>5d} {avg:>10.3f}")


if __name__ == "__main__":
    main()
