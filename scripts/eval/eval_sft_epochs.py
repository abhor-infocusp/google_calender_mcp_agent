#!/usr/bin/env python3
"""Evaluate SFT checkpoints on training data and RL data.

For each checkpoint (epoch):
1. Merge LoRA adapter into fp16
2. Start vLLM server with hermes tool-call parser
3. Run eval on training data (SFT trajectory queries)
4. Run eval on RL data (calendars 0-19)
5. Save results incrementally to JSON + CSV

Resumes from where it left off if interrupted.

Usage:
    PYTHONPATH=src python scripts/eval/eval_sft_epochs.py [--output-dir DIR] [--epochs 1,5,10]
"""

import argparse
import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import glob
import urllib.request

PYTHON = "/home/abhor/miniconda3/envs/agentic/bin/python"
PROJECT_DIR = "/home/abhor/google_calender_mcp_agent"
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "sft_output_100ep")
MERGED_DIR = os.path.join(CHECKPOINT_DIR, "merged_eval")
VLLM_PORT = 8005
MODEL_NAME = "sft-eval"
VLLM_LOG = "/tmp/vllm_epoch_eval.log"

# RL calendars 0-19 (4 and 9 don't exist)
RL_CALENDARS = [i for i in range(20) if i not in (4, 9)]
RL_DATA_DIR = os.path.join(PROJECT_DIR, "rl_data")
SFT_DATA_DIR = os.path.join(PROJECT_DIR, "sft_data")
TRAJ_DIR = os.path.join(SFT_DATA_DIR, "trajectories")


def discover_checkpoints(checkpoint_dir):
    """Find all epoch checkpoints and return sorted by epoch number."""
    pattern = os.path.join(checkpoint_dir, "checkpoint-*")
    ckpts = glob.glob(pattern)
    result = []
    for path in ckpts:
        name = os.path.basename(path)
        step = int(name.split("-")[1])
        result.append((step, path, name))
    result.sort(key=lambda x: x[0])
    return result


def merge_checkpoint(ckpt_path: str):
    """Merge LoRA checkpoint into fp16 model."""
    print(f"  Merging {os.path.basename(ckpt_path)}...", flush=True)
    merge_script = f"""
import os, shutil
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from unsloth import FastLanguageModel

ckpt = "{ckpt_path}"
out = "{MERGED_DIR}"
if os.path.exists(out):
    shutil.rmtree(out)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ckpt, max_seq_length=4096, load_in_4bit=True,
)
model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
print("Merge complete.")
"""
    result = subprocess.run(
        [PYTHON, "-c", merge_script],
        capture_output=True, text=True, timeout=600,
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONPATH": os.path.join(PROJECT_DIR, "src")},
    )
    if result.returncode != 0:
        print(f"  MERGE FAILED: {result.stderr[-500:]}")
        return False
    print("  Merge OK.", flush=True)
    return True


def start_vllm():
    """Start vLLM server and wait for it to be ready."""
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    with open(VLLM_LOG, "w") as logf:
        proc = subprocess.Popen(
            [
                PYTHON, "-m", "vllm.entrypoints.openai.api_server",
                "--model", MERGED_DIR,
                "--served-model-name", MODEL_NAME,
                "--enable-auto-tool-choice",
                "--tool-call-parser", "hermes",
                "--max-model-len", "2048",
                "--gpu-memory-utilization", "0.80",
                "--port", str(VLLM_PORT),
            ],
            stdout=logf, stderr=logf, env=env,
        )

    for attempt in range(60):
        time.sleep(5)
        try:
            resp = urllib.request.urlopen(
                f"http://localhost:{VLLM_PORT}/v1/models", timeout=5
            )
            if resp.status == 200:
                print("  vLLM ready.", flush=True)
                return proc
        except Exception:
            pass
    print("  vLLM FAILED to start!")
    proc.kill()
    return None


def kill_vllm(proc):
    """Kill vLLM server and free GPU."""
    if proc:
        proc.kill()
        proc.wait()
    # Kill any remaining GPU processes from us
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        our_uid = os.getuid()
        for pid_str in result.stdout.strip().split("\n"):
            pid_str = pid_str.strip()
            if not pid_str:
                continue
            try:
                pid = int(pid_str)
                # Only kill our own processes
                stat_path = f"/proc/{pid}/status"
                if os.path.exists(stat_path):
                    with open(stat_path) as f:
                        for line in f:
                            if line.startswith("Uid:"):
                                uid = int(line.split()[1])
                                if uid == our_uid:
                                    os.kill(pid, signal.SIGKILL)
                                break
            except (ProcessLookupError, ValueError, PermissionError):
                pass
    except Exception:
        pass
    time.sleep(3)


def run_eval_calendar(cal_idx, use_sft_data=False, use_rl_data=False, save_path=None):
    """Run eval_qwen.py on a single calendar. Returns (correct, total)."""
    args = [
        PYTHON, os.path.join(PROJECT_DIR, "scripts/eval/eval_qwen.py"),
        str(cal_idx),
        "--model", MODEL_NAME,
        "--base-url", f"http://localhost:{VLLM_PORT}/v1",
        "--with-final-answer",
    ]
    if use_sft_data:
        args.append("--sft-data")
    if use_rl_data:
        args.append("--rl-data")
    if save_path:
        args.extend(["--save", save_path])

    result = subprocess.run(
        args,
        capture_output=True, text=True, timeout=900,
        cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONPATH": os.path.join(PROJECT_DIR, "src")},
    )

    if result.returncode != 0:
        print(f"    Cal {cal_idx} FAILED: {result.stderr[-200:]}")
        return None, None

    # Parse verdicts from output
    clean = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)
    verdicts = re.findall(r'\[EVAL RESULT\]\s+(\w+)', clean)
    correct = sum(1 for v in verdicts if v == "Correct")
    total = len(verdicts)
    return correct, total


def eval_training_data(traj_queries_by_cal):
    """Evaluate on SFT training queries only (matching by query text).

    Instead of running eval_qwen.py (which runs ALL queries for a calendar),
    we run it and then filter to training queries only.
    """
    total_correct = 0
    total_count = 0

    for cal_idx in sorted(traj_queries_by_cal.keys()):
        save_path = f"/tmp/epoch_eval_sft_cal{cal_idx}.json"
        correct, total = run_eval_calendar(cal_idx, use_sft_data=True, save_path=save_path)

        if correct is None:
            print(f"    Train cal {cal_idx}: FAILED")
            continue

        # Filter to only training trajectory queries
        train_queries = traj_queries_by_cal[cal_idx]
        try:
            with open(save_path) as f:
                results = json.load(f)
            train_correct = sum(
                1 for r in results
                if r["query"].strip() in train_queries and r["eval_verdict"] == "Correct"
            )
            train_total = sum(
                1 for r in results if r["query"].strip() in train_queries
            )
        except Exception:
            train_correct = 0
            train_total = 0

        total_correct += train_correct
        total_count += train_total
        pct = f"{train_correct/train_total*100:.0f}%" if train_total else "N/A"
        print(f"    Train cal {cal_idx}: {train_correct}/{train_total} ({pct})")

    return total_correct, total_count


def eval_rl_data(rl_calendars):
    """Evaluate on RL data calendars."""
    total_correct = 0
    total_count = 0

    for cal_idx in rl_calendars:
        save_path = f"/tmp/epoch_eval_rl_cal{cal_idx}.json"
        correct, total = run_eval_calendar(cal_idx, use_rl_data=True, save_path=save_path)

        if correct is None:
            print(f"    RL cal {cal_idx}: FAILED")
            continue

        total_correct += correct
        total_count += total
        pct = f"{correct/total*100:.0f}%" if total else "N/A"
        print(f"    RL cal {cal_idx}: {correct}/{total} ({pct})")

    return total_correct, total_count


def load_training_queries():
    """Load the SFT training trajectory queries grouped by calendar index."""
    queries_by_cal = {}
    for f in sorted(glob.glob(os.path.join(TRAJ_DIR, "*.json"))):
        cal = int(os.path.basename(f).replace(".json", ""))
        with open(f) as fh:
            data = json.load(fh)
        queries_by_cal[cal] = set(t["query"].strip() for t in data)
    return queries_by_cal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(CHECKPOINT_DIR, "eval_results"))
    parser.add_argument("--epochs", default=None, help="Comma-separated epoch steps to eval (default: all)")
    parser.add_argument("--checkpoint-dir", default=CHECKPOINT_DIR)
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    results_json = os.path.join(output_dir, "epoch_eval_results.json")
    results_csv = os.path.join(output_dir, "epoch_eval_results.csv")

    # Load existing results for resume
    existing_results = {}
    if os.path.exists(results_json):
        with open(results_json) as f:
            existing_results = json.load(f)
        print(f"Loaded {len(existing_results)} existing results (resuming)")

    # Discover checkpoints
    checkpoints = discover_checkpoints(args.checkpoint_dir)
    if args.epochs:
        selected_steps = set(int(s) for s in args.epochs.split(","))
        checkpoints = [c for c in checkpoints if c[0] in selected_steps]

    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    if not checkpoints:
        print("No checkpoints found. Run training first.")
        return

    # Load training queries for filtering
    traj_queries_by_cal = load_training_queries()
    total_train_queries = sum(len(v) for v in traj_queries_by_cal.values())
    print(f"Training queries: {total_train_queries} across {len(traj_queries_by_cal)} calendars")
    print(f"RL calendars: {RL_CALENDARS}")

    # Init CSV
    if not os.path.exists(results_csv):
        with open(results_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "checkpoint", "step", "train_correct", "train_total", "train_acc",
                "rl_correct", "rl_total", "rl_acc",
            ])

    for step, ckpt_path, ckpt_name in checkpoints:
        if ckpt_name in existing_results:
            print(f"\n{'='*60}")
            print(f"SKIP {ckpt_name} (already evaluated)")
            continue

        print(f"\n{'='*60}")
        print(f"EVALUATING: {ckpt_name} (step {step})")
        print(f"{'='*60}")

        # Merge
        if not merge_checkpoint(ckpt_path):
            existing_results[ckpt_name] = {"step": step, "error": "merge_failed"}
            continue

        # Start vLLM
        proc = start_vllm()
        if not proc:
            existing_results[ckpt_name] = {"step": step, "error": "vllm_failed"}
            continue

        try:
            # Eval training data
            print("\n  --- Training Data Eval ---")
            train_correct, train_total = eval_training_data(traj_queries_by_cal)
            train_acc = train_correct / train_total if train_total else 0

            # Eval RL data
            print("\n  --- RL Data Eval (cals 0-19) ---")
            rl_correct, rl_total = eval_rl_data(RL_CALENDARS)
            rl_acc = rl_correct / rl_total if rl_total else 0

            result = {
                "step": step,
                "train_correct": train_correct,
                "train_total": train_total,
                "train_acc": round(train_acc, 4),
                "rl_correct": rl_correct,
                "rl_total": rl_total,
                "rl_acc": round(rl_acc, 4),
            }
            existing_results[ckpt_name] = result

            print(f"\n  RESULTS: train={train_correct}/{train_total} ({train_acc:.1%}), "
                  f"rl={rl_correct}/{rl_total} ({rl_acc:.1%})")

            # Append to CSV
            with open(results_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ckpt_name, step, train_correct, train_total, f"{train_acc:.4f}",
                    rl_correct, rl_total, f"{rl_acc:.4f}",
                ])

        except Exception as e:
            print(f"  ERROR: {e}")
            existing_results[ckpt_name] = {"step": step, "error": str(e)}

        finally:
            kill_vllm(proc)

        # Save results incrementally
        with open(results_json, "w") as f:
            json.dump(existing_results, f, indent=2)

    # Final summary
    print(f"\n{'='*60}")
    print("EPOCH EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<25} {'Step':>6} {'Train':>12} {'RL':>12}")
    print("-" * 60)
    for ckpt_name, r in sorted(existing_results.items(), key=lambda x: x[1].get("step", 0)):
        if "error" in r:
            print(f"{ckpt_name:<25} {r['step']:>6} ERROR: {r['error']}")
        else:
            t = f"{r['train_correct']}/{r['train_total']} ({r['train_acc']:.1%})"
            e = f"{r['rl_correct']}/{r['rl_total']} ({r['rl_acc']:.1%})"
            print(f"{ckpt_name:<25} {r['step']:>6} {t:>12} {e:>12}")

    print(f"\nResults: {results_json}")
    print(f"CSV:     {results_csv}")


if __name__ == "__main__":
    main()
