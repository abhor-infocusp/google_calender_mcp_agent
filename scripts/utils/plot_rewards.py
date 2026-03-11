#!/usr/bin/env python3
"""Plot training and validation reward curves and save to reward_curve.png."""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

from calendar_agent.paths import PROJECT_ROOT
TRAJ_DIR = str(PROJECT_ROOT / ".art" / "calendar-agent" / "models" / "calendar-agent-001" / "trajectories")


def load_step_rewards(subdir):
    """Load (step, avg_reward) pairs from trajectory JSONL files."""
    pattern = os.path.join(TRAJ_DIR, subdir, "*.jsonl")
    results = []
    for f in sorted(glob.glob(pattern)):
        step = int(os.path.basename(f).replace(".jsonl", ""))
        rewards = []
        with open(f) as fh:
            for line in fh:
                data = json.loads(line)
                for t in data.get("trajectories", []):
                    rewards.append(t.get("reward", 0.0))
        if rewards:
            results.append((step, np.mean(rewards)))
    return results


def moving_average(values, window):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    train_data = load_step_rewards("train")
    val_data = load_step_rewards("val")

    train_steps = [s for s, _ in train_data]
    train_rewards = [r for _, r in train_data]
    val_steps = [s for s, _ in val_data]
    val_rewards = [r for _, r in val_data]

    window = 20
    train_ma = moving_average(train_rewards, window)
    train_ma_steps = train_steps[window - 1 :]

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.scatter(train_steps, train_rewards, alpha=0.15, s=8, color="tab:blue", label="Train (per step)")
    ax.plot(train_ma_steps, train_ma, color="tab:blue", linewidth=2, label=f"Train (MA-{window})")
    ax.plot(val_steps, val_rewards, "o-", color="tab:orange", markersize=5, linewidth=1.5, label="Validation")

    ax.set_xlabel("Step")
    ax.set_ylabel("Avg Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("reward_curve.png", dpi=150)
    print(f"Saved reward_curve.png ({len(train_data)} train steps, {len(val_data)} val steps)")


if __name__ == "__main__":
    main()
