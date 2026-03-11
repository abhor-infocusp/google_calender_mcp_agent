#!/usr/bin/env python3
"""Compact ART training results viewer with flag-based querying.

Usage:
  python view_results.py                  # show available sections
  python view_results.py summary          # training summary stats
  python view_results.py table            # step-by-step metrics table
  python view_results.py trajs            # recent trajectory analysis (last 3 train + last val)
  python view_results.py all              # all of the above
  python view_results.py step 18          # trajectories for a specific train step
  python view_results.py val 20           # trajectories for a specific val step
  python view_results.py step 18 --full   # full conversation for a specific step
"""

import json
import sys
from pathlib import Path

ART_DIR = Path(__file__).parent / ".art" / "calendar-agent" / "models" / "calendar-agent-001"
HISTORY_FILE = ART_DIR / "history.jsonl"
TRAJECTORIES_DIR = ART_DIR / "trajectories"


def load_history():
    if not HISTORY_FILE.exists():
        print(f"No history file at {HISTORY_FILE}")
        sys.exit(1)
    entries = []
    with open(HISTORY_FILE) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def load_steps(entries):
    steps = {}
    for entry in entries:
        s = entry["step"]
        if s not in steps:
            steps[s] = {}
        steps[s].update(entry)
    return steps


def load_trajectories(split, step):
    path = TRAJECTORIES_DIR / split / f"{step:04d}.jsonl"
    if not path.exists():
        return []
    trajs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if "trajectories" in entry:
                    trajs.extend(entry["trajectories"])
                else:
                    trajs.append(entry)
    return trajs


def analyze_trajectory(traj):
    reward = traj.get("reward", 0)
    metrics = traj.get("metrics", {})
    metadata = traj.get("metadata", {})
    messages = traj.get("messages_and_choices", [])

    tool_calls = 0
    tool_names = []
    final_answer = None
    user_query = None
    for msg in messages:
        if msg.get("role") == "user" and user_query is None:
            user_query = msg.get("content", "")
        if isinstance(msg, dict) and "message" in msg:
            inner = msg["message"]
            if inner.get("tool_calls"):
                for tc in inner["tool_calls"]:
                    tool_calls += 1
                    fn = tc.get("function", {}).get("name", "?")
                    tool_names.append(fn)
            content = inner.get("content", "")
            if content:
                if "</think>" in content:
                    content = content.split("</think>", 1)[1].strip()
                if content and not content.startswith("<think>"):
                    final_answer = content

    return {
        "reward": reward,
        "correct": metrics.get("correct", 0),
        "verdict": metrics.get("verdict", 0),
        "tokens": metrics.get("completion_tokens", 0),
        "exception": metrics.get("exception_rate", 0) > 0 or metadata.get("exception", False),
        "tool_calls": tool_calls,
        "tool_names": tool_names,
        "scenario": metadata.get("scenario_id", "?"),
        "complexity": metadata.get("complexity", "?"),
        "query": user_query,
        "answer": final_answer,
        "answer_short": (final_answer[:80] + "...") if final_answer and len(final_answer) > 80 else final_answer,
    }


def extract_full_conversation(traj):
    """Extract the full message sequence from a trajectory."""
    messages = traj.get("messages_and_choices", [])
    conv = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            conv.append(("SYSTEM", msg.get("content", "")[:120] + "..."))
        elif role == "user":
            conv.append(("USER", msg.get("content", "")))
        elif role == "tool":
            content = msg.get("content", "")
            conv.append(("TOOL", content[:200] + ("..." if len(content) > 200 else "")))
        elif isinstance(msg, dict) and "message" in msg:
            inner = msg["message"]
            content = inner.get("content", "")
            if "</think>" in content:
                content = content.split("</think>", 1)[1].strip()
            tcs = inner.get("tool_calls", [])
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {})
                    conv.append(("CALL", f"{fn.get('name', '?')}({fn.get('arguments', '')[:120]})"))
            if content:
                conv.append(("ASSISTANT", content))
    return conv


# --- Print functions ---

def print_summary(steps):
    max_step = max(steps.keys())
    train_rewards = [(s, d["train/reward"]) for s, d in sorted(steps.items()) if "train/reward" in d]
    val_rewards = [(s, d["val/reward"]) for s, d in sorted(steps.items()) if "val/reward" in d]
    trained_steps = [(s, d.get("train/num_groups_trainable", 0)) for s, d in sorted(steps.items()) if "train/num_groups_trainable" in d]
    actually_trained = [s for s, n in trained_steps if n > 0]

    print(f"=== Summary (steps 0–{max_step}) ===")
    print(f"GRPO updates: {len(actually_trained)}/{len(trained_steps)} steps — {actually_trained}")
    if train_rewards:
        rewards_only = [r for _, r in train_rewards]
        nonzero = [r for r in rewards_only if r > 0]
        if nonzero:
            print(f"Train reward: avg={sum(rewards_only)/len(rewards_only):.3f}, nonzero avg={sum(nonzero)/len(nonzero):.3f} ({len(nonzero)}/{len(rewards_only)} steps)")
        else:
            print(f"Train reward: all zero across {len(rewards_only)} steps")
    if val_rewards:
        for s, r in val_rewards:
            print(f"Val step {s}: reward={r:.3f}")


def print_table(steps):
    print(f"\n{'Step':>4} {'Reward':>7} {'Grps':>5} {'Tokens':>7} {'Verdict':>8}")
    print("-" * 36)
    for s in sorted(steps):
        d = steps[s]
        r = d.get("train/reward")
        if r is None:
            continue
        t = d.get("train/num_groups_trainable", "-")
        tok = d.get("train/completion_tokens", 0)
        v = d.get("train/verdict", 0)
        marker = " *" if (isinstance(t, int) and t > 0) else ""
        print(f"{s:>4} {r:>7.3f} {str(t):>5} {tok:>7.0f} {v:>8.3f}{marker}")


def print_traj_summary(split, step):
    """Print compact trajectory summary for a step."""
    trajs = load_trajectories(split, step)
    if not trajs:
        print(f"No {split} trajectories for step {step}")
        return

    analyzed = [analyze_trajectory(t) for t in trajs]
    n_correct = sum(1 for a in analyzed if a["reward"] > 0)
    n_tools = sum(1 for a in analyzed if a["tool_calls"] > 0)
    n_exc = sum(1 for a in analyzed if a["exception"])
    avg_tokens = sum(a["tokens"] for a in analyzed) / len(analyzed) if analyzed else 0
    avg_tools = sum(a["tool_calls"] for a in analyzed) / len(analyzed) if analyzed else 0

    print(f"--- {split} step {step}: {len(trajs)} rollouts, {n_correct} correct, {n_tools} used tools, {n_exc} exc, avg {avg_tokens:.0f} tok, avg {avg_tools:.1f} calls ---")

    by_scenario = {}
    for a in analyzed:
        by_scenario.setdefault(a["scenario"], []).append(a)

    for scenario, group in by_scenario.items():
        rewards = [a["reward"] for a in group]
        cx = group[0]["complexity"]
        n_ok = sum(1 for r in rewards if r > 0)
        tools = [a["tool_calls"] for a in group]
        print(f"  {scenario} ({cx}): {n_ok}/{len(group)} correct, tools={tools}, rewards={rewards}")

        correct_ex = next((a for a in group if a["reward"] > 0), None)
        wrong_ex = next((a for a in group if a["reward"] == 0 and not a["exception"]), None)
        if correct_ex and correct_ex["answer_short"]:
            print(f"    OK: {correct_ex['answer_short']}")
        if wrong_ex and wrong_ex["answer_short"]:
            print(f"    FAIL: {wrong_ex['answer_short']}")


def print_traj_full(split, step):
    """Print full conversations for a step's trajectories."""
    trajs = load_trajectories(split, step)
    if not trajs:
        print(f"No {split} trajectories for step {step}")
        return

    for i, traj in enumerate(trajs):
        a = analyze_trajectory(traj)
        print(f"\n== Rollout {i+1}/{len(trajs)} | {a['scenario']} ({a['complexity']}) | reward={a['reward']} | tools={a['tool_calls']} ==")
        conv = extract_full_conversation(traj)
        for role, content in conv:
            if role == "SYSTEM":
                continue
            print(f"  [{role}] {content[:300]}{'...' if len(content) > 300 else ''}")


def print_recent_trajs():
    """Print last 3 train + last val step trajectories."""
    for split in ["train", "val"]:
        split_dir = TRAJECTORIES_DIR / split
        if not split_dir.exists():
            continue
        files = sorted(split_dir.glob("*.jsonl"))
        if not files:
            continue
        show = files[-3:] if split == "train" else files[-1:]
        for f in show:
            print_traj_summary(split, int(f.stem))


def print_help():
    print("""view_results.py — ART training results viewer

Commands:
  summary          Training summary (GRPO updates, reward stats, val scores)
  table            Per-step metrics table (reward, groups trained, tokens, verdict)
  trajs            Recent trajectories (last 3 train + last val)
  all              All of the above
  step <N>         Trajectory details for train step N
  val <N>          Trajectory details for val step N
  step <N> --full  Full conversation dump for train step N
  val <N> --full   Full conversation dump for val step N

Available trajectory steps:""")
    for split in ["train", "val"]:
        split_dir = TRAJECTORIES_DIR / split
        if split_dir.exists():
            files = sorted(split_dir.glob("*.jsonl"))
            step_nums = [int(f.stem) for f in files]
            print(f"  {split}: {step_nums}")


def main():
    args = sys.argv[1:]

    if not args:
        print_help()
        return

    cmd = args[0]

    if cmd == "help":
        print_help()
    elif cmd == "summary":
        print_summary(load_steps(load_history()))
    elif cmd == "table":
        print_table(load_steps(load_history()))
    elif cmd == "trajs":
        print_recent_trajs()
    elif cmd == "all":
        steps = load_steps(load_history())
        print_summary(steps)
        print_table(steps)
        print()
        print_recent_trajs()
        print("\n=== END ===")
    elif cmd in ("step", "val"):
        if len(args) < 2:
            print(f"Usage: view_results.py {cmd} <step_number> [--full]")
            return
        step_num = int(args[1])
        split = "train" if cmd == "step" else "val"
        full = "--full" in args
        if full:
            print_traj_full(split, step_num)
        else:
            print_traj_summary(split, step_num)
    else:
        print(f"Unknown command: {cmd}")
        print_help()


if __name__ == "__main__":
    main()
