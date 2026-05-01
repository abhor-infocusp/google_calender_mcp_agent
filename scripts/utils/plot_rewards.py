#!/usr/bin/env python3
"""Plot RL training curves from ART trajectory parquets and history.jsonl.

Single combined figure: overall reward, overall reward slope, per-category
reward, per-category reward slope, skip rate.

Usage:
    # Plot any run by run-dir (auto-finds .art/<project>/models/<name>/):
    plot_rewards.py --run-dir runs/rl_grpo_qwen3_14b_base_20260426
    plot_rewards.py --run-dir runs/rl_grpo_qwen3_14b_base_20260426 \
                    --output ./out.png --title "GRPO from base"

    # Direct ART model path:
    plot_rewards.py --art-dir .art/grpo-base-20260426/models/qwen3-14b-base

    # No args: legacy default (the original 2026-04-20 calendar-agent run).
"""

import argparse
import json
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calendar_agent.paths import PROJECT_ROOT

WINDOW = 300  # MA window for reward smoothing
DERIV_SMOOTH = 200  # Second MA pass on the derivative


def find_art_model_dir(run_dir: str) -> str:
    """Locate the ART model dir for a run.

    ART writes to `.art/<project>/models/<name>/`. Depending on cwd at launch,
    this may be at the repo root OR nested inside `<run_dir>/.art/...`.
    Read the trainer's startup log line for the exact project + name so we
    don't mis-match across concurrent runs that share the repo `.art/`.
    """
    import re
    # Find project + name from the trainer's startup log.
    train_logs = sorted(glob.glob(os.path.join(run_dir, "logs", "train_*.log")))
    project = None
    name = None
    pat = re.compile(r"project=(\S+)\s+name=(\S+)")
    for log in reversed(train_logs):
        try:
            with open(log) as f:
                # The "[rl_train] base_model=... project=... name=..." line
                # is in the first ~50 lines after vLLM startup banner.
                for _ in range(2000):
                    line = f.readline()
                    if not line:
                        break
                    m = pat.search(line)
                    if m:
                        project, name = m.group(1), m.group(2)
                        break
            if project:
                break
        except Exception:
            continue

    candidates = []
    if project and name:
        for base in [run_dir, str(PROJECT_ROOT)]:
            cand = os.path.join(base, ".art", project, "models", name)
            candidates.append(cand)
    else:
        # Fallback: glob and pick the most recently-modified one (best-effort).
        for base in [run_dir, str(PROJECT_ROOT)]:
            for p in sorted(glob.glob(os.path.join(base, ".art/*/models/*/"))):
                candidates.append(p.rstrip("/"))

    for c in candidates:
        if os.path.exists(os.path.join(c, "history.jsonl")):
            return c
    raise FileNotFoundError(
        f"no .art/<project>/models/<name>/history.jsonl found for {run_dir} "
        f"(tried project={project!r} name={name!r})"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="runs/<run>/ dir; auto-finds ART model dir")
    p.add_argument("--art-dir", help="explicit path to .art/<project>/models/<name>/")
    p.add_argument("--output", help="output PNG path (default: <run-dir>/reward_curve.png or ./reward_curve.png)")
    p.add_argument("--title", help="plot title (default: derived from run-dir)")
    return p.parse_args()


# Default values for legacy invocation (no args). Resolved at runtime in main().
ART_DIR = ""
TRAJ_DIR = ""
HISTORY_PATH = ""

CATEGORY_SHORT = {
    "Complex Logic & Conflict (Advanced)": "Complex",
    "Human Chaos (Edge Cases/Fragments)": "Chaos",
    "Information Retrieval (Querying)": "IR",
    "Modifier & Correction (Rescheduling/Updates)": "Modifier",
    "Relative Time References (today, tomorrow, yesterday, this week)": "RelTime",
    "Schedule a Single Event": "Schedule",
    "Vague & Contextual (Reasoning Required)": "Vague",
}

CATEGORY_COLORS = {
    "Complex": "#e41a1c",
    "Chaos": "#ff7f00",
    "IR": "#4daf4a",
    "Modifier": "#377eb8",
    "RelTime": "#984ea3",
    "Schedule": "#a65628",
    "Vague": "#f781bf",
}


def load_history(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_category_rewards():
    """Load per-step per-category rewards from trajectory parquets."""
    category_data = defaultdict(list)
    files = sorted(glob.glob(os.path.join(TRAJ_DIR, "*.parquet")))
    for f in files:
        step = int(os.path.basename(f).replace(".parquet", ""))
        df = pd.read_parquet(f, columns=["reward", "metadata"])
        for _, row in df.iterrows():
            meta = row["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            cat = CATEGORY_SHORT.get(meta.get("category", "Unknown"), "Unk")
            category_data[cat].append((step, row["reward"]))
    return category_data


def moving_average(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(np.mean(values[start : i + 1]))
    return out


def smooth_derivative(y, ma_window=DERIV_SMOOTH, scale=100):
    """Compute a smoothed derivative: slope per `scale` steps."""
    arr = np.array(y)
    deriv = np.gradient(arr) * scale
    return np.array(moving_average(deriv.tolist(), ma_window))


def main():
    global ART_DIR, TRAJ_DIR, HISTORY_PATH
    args = parse_args()

    if args.art_dir:
        ART_DIR = args.art_dir.rstrip("/")
    elif args.run_dir:
        ART_DIR = find_art_model_dir(args.run_dir)
    else:
        # Legacy default — the 2026-04-20 main RL run.
        ART_DIR = str(PROJECT_ROOT / ".art" / "calendar-agent" / "models" / "calendar-agent-001")
    TRAJ_DIR = os.path.join(ART_DIR, "trajectories", "train")
    HISTORY_PATH = os.path.join(ART_DIR, "history.jsonl")

    if args.output:
        output_path = args.output
    elif args.run_dir:
        output_path = os.path.join(args.run_dir, "reward_curve.png")
    else:
        output_path = "reward_curve.png"

    if args.title:
        title = args.title
    elif args.run_dir:
        title = f"RL Training — {os.path.basename(args.run_dir.rstrip('/'))}"
    else:
        title = "RL Training — Qwen3-14B GRPO"

    print(f"ART_DIR: {ART_DIR}")
    print(f"output:  {output_path}")
    print(f"title:   {title}")

    if not os.path.exists(HISTORY_PATH):
        print(f"ERROR: history.jsonl not found at {HISTORY_PATH}", file=sys.stderr)
        sys.exit(1)

    history = load_history(HISTORY_PATH)
    print(f"Loaded {len(history)} steps from history.jsonl")

    steps = np.array([r["step"] for r in history])
    rewards = [r["train/reward"] for r in history]
    trainable = [r.get("data/step_num_groups_trainable", 0) for r in history]
    skipped = [1 if t == 0 else 0 for t in trainable]

    reward_ma = moving_average(rewards, WINDOW)
    skip_ma = moving_average(skipped, WINDOW)
    reward_slope = smooth_derivative(reward_ma)

    # Per-category reward MAs + slopes (indexed by training step)
    category_data = load_category_rewards()
    print(f"Found {len(category_data)} categories")

    cat_steps = {}  # cat -> list of steps that sampled this category
    cat_ma = {}     # cat -> MA of rewards at those steps
    cat_slope = {}  # cat -> smoothed derivative (per-step in cat-step-space)
    cat_summary = {}

    for cat, data in category_data.items():
        step_rewards = defaultdict(list)
        for s, r in data:
            step_rewards[s].append(r)
        sorted_steps = sorted(step_rewards.keys())
        means = [float(np.mean(step_rewards[s])) for s in sorted_steps]
        # Each category sampled ~1 in every 7 steps; use smaller MA in cat-index space.
        # Doubled from WINDOW//7 for heavier smoothing of per-category reward.
        cat_window = max(60, (WINDOW // 7) * 2)
        ma = moving_average(means, cat_window)
        # Derivative: slope per 100 training steps.
        # We have cat-step-indexed rewards; scale the gradient by step-spacing.
        if len(sorted_steps) >= 2:
            step_spacing = (sorted_steps[-1] - sorted_steps[0]) / max(1, len(sorted_steps) - 1)
            # gradient per cat-index * (cat-index / 100 training steps) = gradient per 100 steps
            scale = 100.0 / step_spacing
        else:
            scale = 1.0
        # Doubled from DERIV_SMOOTH//7 for heavier smoothing of per-category slope.
        slope = smooth_derivative(ma, ma_window=max(40, (DERIV_SMOOTH // 7) * 2), scale=scale)
        cat_steps[cat] = sorted_steps
        cat_ma[cat] = ma
        cat_slope[cat] = slope
        cat_summary[cat] = {"n_rollouts": len(data), "mean_acc": float(np.mean(means))}

    # ── Build combined figure ──
    fig, axes = plt.subplots(
        5, 1, figsize=(14, 16), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 3, 2, 1.5]},
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")

    # 1) Overall reward
    ax = axes[0]
    ax.scatter(steps, rewards, alpha=0.15, s=6, color="steelblue", label="Per-step")
    ax.plot(steps, reward_ma, color="navy", linewidth=2.2, label=f"MA-{WINDOW}")
    ax.set_ylabel("Overall Reward")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="major", alpha=0.6, linewidth=0.8, color="gray")
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.5, color="gray", linestyle=":")
    ax.minorticks_on()

    # 2) Overall reward slope
    ax = axes[1]
    stable = WINDOW  # skip the warm-up region
    ax.plot(steps[stable:], reward_slope[stable:], color="darkorange", linewidth=2.0,
            label="d(reward)/d(step) × 100 (smoothed)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Overall Slope\n(Δreward / 100 steps)")
    vals = reward_slope[stable:]
    if len(vals):
        m = max(abs(vals.min()), abs(vals.max())) * 1.3
        ax.set_ylim(-m, m)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="major", alpha=0.6, linewidth=0.8, color="gray")
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.5, color="gray", linestyle=":")
    ax.minorticks_on()

    # 3) Per-category reward
    ax = axes[2]
    for cat in sorted(cat_ma.keys()):
        color = CATEGORY_COLORS.get(cat, "gray")
        acc = cat_summary[cat]["mean_acc"]
        ax.plot(cat_steps[cat], cat_ma[cat], linewidth=1.8, color=color,
                label=f"{cat} ({acc:.0%})")
    ax.set_ylabel("Per-Category Reward")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=8, ncol=4)
    ax.grid(True, which="major", alpha=0.6, linewidth=0.8, color="gray")
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.5, color="gray", linestyle=":")
    ax.minorticks_on()

    # 4) Per-category reward slope — skip first 500 training steps
    # (warm-up has huge swings that compress the y-axis)
    ax = axes[3]
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    SLOPE_START = 500
    all_slopes = []
    for cat in sorted(cat_slope.keys()):
        color = CATEGORY_COLORS.get(cat, "gray")
        s_arr = cat_slope[cat]
        s_steps = np.array(cat_steps[cat])
        mask = s_steps >= SLOPE_START
        ax.plot(s_steps[mask], s_arr[mask], linewidth=1.5, color=color, label=cat)
        all_slopes.extend(s_arr[mask].tolist())
    ax.set_ylabel("Per-Category Slope\n(Δreward / 100 steps)")
    if all_slopes:
        m = max(abs(min(all_slopes)), abs(max(all_slopes))) * 1.2
        ax.set_ylim(-m, m)
    ax.legend(loc="upper right", fontsize=8, ncol=4)
    ax.grid(True, which="major", alpha=0.6, linewidth=0.8, color="gray")
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.5, color="gray", linestyle=":")
    ax.minorticks_on()

    # 5) Skip rate
    ax = axes[4]
    ax.plot(steps, skip_ma, color="darkred", linewidth=2.0, label=f"Skip rate (MA-{WINDOW})")
    ax.set_ylabel("Skip Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Training Step")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="major", alpha=0.6, linewidth=0.8, color="gray")
    ax.grid(True, which="minor", alpha=0.3, linewidth=0.5, color="gray", linestyle=":")
    ax.minorticks_on()

    # Stats footer
    total_trained = sum(1 for t in trainable if t > 0)
    total_skipped = sum(skipped)
    stats = (
        f"Steps: {len(history)} | Trained: {total_trained} | Skipped: {total_skipped} "
        f"({total_skipped / len(history) * 100:.0f}%) | "
        f"Mean reward: {np.mean(rewards):.3f}"
    )
    fig.text(0.5, 0.005, stats, ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved {output_path}")

    # Summary table
    print(f"\n{'Category':<12s} {'Rollouts':>9s} {'Accuracy':>9s}")
    print("-" * 35)
    for cat in sorted(cat_summary.keys()):
        s = cat_summary[cat]
        print(f"{cat:<12s} {s['n_rollouts']:>9d} {s['mean_acc']:>8.1%}")


if __name__ == "__main__":
    main()
