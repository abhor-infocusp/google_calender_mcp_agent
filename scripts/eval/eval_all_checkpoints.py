#!/usr/bin/env python3
"""Evaluate SFT checkpoints on RL data: merge LoRA, serve via vLLM, run eval.

Discovers checkpoints dynamically from sft_output/checkpoint-*.
Skips already-evaluated ones (checks sft_output/eval/checkpoint-{N}.json).
Run manually whenever new checkpoints appear from training.

Results:
    sft_output/eval/checkpoint-{N}.json  — per-checkpoint detailed results
    sft_output/eval/summary.csv          — one row per checkpoint

Usage:
    PYTHONPATH=src python scripts/eval/eval_all_checkpoints.py
"""

import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from calendar_agent.paths import PROJECT_ROOT, SFT_OUTPUT_DIR

PYTHON = sys.executable
PROJECT = str(PROJECT_ROOT)
SFT_OUTPUT = str(SFT_OUTPUT_DIR)
MERGED_DIR = os.path.join(SFT_OUTPUT, "merged_tmp")
EVAL_DIR = os.path.join(SFT_OUTPUT, "eval")
SUMMARY_CSV = os.path.join(EVAL_DIR, "summary.csv")
LOSS_CSV = os.path.join(SFT_OUTPUT, "epoch_losses.csv")
PORT = 8005

# Short names for summary table columns (same order as sorted full names)
CATEGORY_ORDER = [
    "Complex Logic & Conflict (Advanced)",
    "Human Chaos (Edge Cases/Fragments)",
    "Information Retrieval (Querying)",
    "Modifier & Correction (Rescheduling/Updates)",
    "Relative Time References (today, tomorrow, yesterday, this week)",
    "Schedule a Single Event",
    "Vague & Contextual (Reasoning Required)",
]
CATEGORY_SHORT = {
    "Complex Logic & Conflict (Advanced)": "Complex",
    "Human Chaos (Edge Cases/Fragments)": "Chaos",
    "Information Retrieval (Querying)": "IR",
    "Modifier & Correction (Rescheduling/Updates)": "Modifier",
    "Relative Time References (today, tomorrow, yesterday, this week)": "RelTime",
    "Schedule a Single Event": "Schedule",
    "Vague & Contextual (Reasoning Required)": "Vague",
}


# ── Discovery & metadata ─────────────────────────────────────

def discover_checkpoints():
    """Find all checkpoint-* dirs, return sorted list of (step, path)."""
    checkpoints = []
    for name in os.listdir(SFT_OUTPUT):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            step = int(m.group(1))
            path = os.path.join(SFT_OUTPUT, name)
            checkpoints.append((step, path))
    return sorted(checkpoints)


def get_checkpoint_epoch(ckpt_path):
    """Read epoch from trainer_state.json."""
    state_path = os.path.join(ckpt_path, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        return state.get("epoch", None)
    return None


def read_epoch_losses():
    """Read epoch_losses.csv, return {epoch: (train_loss, eval_loss)}."""
    losses = {}
    if not os.path.exists(LOSS_CSV):
        return losses
    with open(LOSS_CSV) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("epoch"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    epoch = int(parts[0])
                    train_loss = float(parts[1])
                    eval_loss = float(parts[2])
                    losses[epoch] = (train_loss, eval_loss)
                except ValueError:
                    continue
    return losses


def is_evaluated(step):
    """Check if checkpoint already has eval results."""
    return os.path.exists(os.path.join(EVAL_DIR, f"checkpoint-{step}.json"))


def load_eval_result(step):
    """Load saved eval result for a checkpoint."""
    path = os.path.join(EVAL_DIR, f"checkpoint-{step}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def category_breakdown(results):
    """Compute per-category accuracy from result list."""
    cats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        cat = r["category"]
        cats[cat]["total"] += 1
        if r["verdict"] == "Correct":
            cats[cat]["correct"] += 1
    breakdown = {}
    for cat in sorted(cats):
        c, t = cats[cat]["correct"], cats[cat]["total"]
        breakdown[cat] = {"correct": c, "total": t, "pct": round(c / t * 100, 1) if t > 0 else 0}
    return breakdown


# ── vLLM & eval operations ───────────────────────────────────

def kill_vllm():
    """Kill any vLLM process on PORT and free GPU memory."""
    subprocess.run(f"lsof -ti :{PORT} | xargs -r kill -9", shell=True, capture_output=True)
    time.sleep(2)
    result = subprocess.run(
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader",
        shell=True, capture_output=True, text=True,
    )
    for line in result.stdout.strip().split("\n"):
        pid = line.strip()
        if not pid:
            continue
        owner = subprocess.run(f"ps -p {pid} -o user=", shell=True, capture_output=True, text=True)
        if "abhor" in owner.stdout:
            subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True)
    time.sleep(3)


def merge_checkpoint(ckpt_num):
    """Merge LoRA checkpoint into fp16 model."""
    ckpt_path = os.path.join(SFT_OUTPUT, f"checkpoint-{ckpt_num}")
    print(f"Merging checkpoint-{ckpt_num}...")

    merge_script = f"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{ckpt_path}",
    max_seq_length=4096,
    load_in_4bit=True,
)
output = "{MERGED_DIR}"
model.save_pretrained_merged(output, tokenizer, save_method="merged_16bit")
print("Merge complete")
"""
    script_path = os.path.join(SFT_OUTPUT, "_merge_tmp.py")
    with open(script_path, "w") as f:
        f.write(merge_script)

    result = subprocess.run(
        [PYTHON, script_path],
        env={**os.environ, "PYTHONPATH": os.path.join(PROJECT, "src"),
             "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1"},
        timeout=300,
    )
    os.remove(script_path)

    if result.returncode != 0:
        print(f"MERGE FAILED (exit code {result.returncode})")
        return False
    print("Merge OK")
    return True


def start_vllm():
    """Start vLLM server and wait for it to be ready."""
    vllm_log = os.path.join(EVAL_DIR, "vllm.log")
    print(f"Starting vLLM on port {PORT}... (log: {vllm_log})")
    vllm_log_fh = open(vllm_log, "w")
    proc = subprocess.Popen(
        [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
         "--model", MERGED_DIR,
         "--served-model-name", "sft-v2",
         "--enable-auto-tool-choice",
         "--tool-call-parser", "hermes",
         "--max-model-len", "4096",
         "--gpu-memory-utilization", "0.70",
         "--port", str(PORT)],
        env={**os.environ, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
        stdout=vllm_log_fh, stderr=subprocess.STDOUT,
    )

    for i in range(120):
        time.sleep(3)
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
            if req.status == 200:
                print("vLLM ready!")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            vllm_log_fh.close()
            out = open(vllm_log).read()
            print(f"vLLM exited early:\n{out[-500:]}")
            return None

    print("vLLM failed to start in 360s")
    proc.kill()
    return None


def run_eval(save_path, num_calendars=20):
    """Run eval_batch.py on RL data (Gemini judge) and return results."""
    cmd = [PYTHON, "-u", os.path.join(PROJECT, "scripts/eval/eval_batch.py"),
         "--mode", "rl",
         "--model", "sft-v2",
         "--base-url", f"http://localhost:{PORT}/v1",
         "--num-calendars", str(num_calendars),
         "--max-queries", "0",
         "--save", save_path]
    subprocess.run(
        cmd,
        env={**os.environ, "PYTHONPATH": os.path.join(PROJECT, "src"),
             "PYTHONUNBUFFERED": "1"},
        timeout=14400,
    )
    if os.path.exists(save_path):
        with open(save_path) as f:
            return json.load(f)
    return None


# ── Results I/O ───────────────────────────────────────────────

def write_summary(checkpoints, losses):
    """Write summary.csv from all evaluated checkpoint JSONs."""
    header = ["checkpoint", "epoch", "train_loss", "eval_loss", "correct", "total", "pct"]
    for cat in CATEGORY_ORDER:
        header.append(CATEGORY_SHORT[cat])

    rows = []
    for step, _ in checkpoints:
        data = load_eval_result(step)
        if not data:
            continue
        rl = data.get("rl", {})
        correct = rl.get("correct", 0)
        total = rl.get("total", 0)
        pct = round(correct / total * 100, 1) if total > 0 else 0
        by_cat = rl.get("by_category", {})

        row = [step, data.get("epoch", ""), data.get("train_loss", ""),
               data.get("eval_loss", ""), correct, total, pct]
        for cat in CATEGORY_ORDER:
            cat_data = by_cat.get(cat, {})
            c, t = cat_data.get("correct", 0), cat_data.get("total", 0)
            row.append(f"{c}/{t}" if t > 0 else "")
        rows.append(row)

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def print_summary(checkpoints, losses):
    """Print summary table to stdout."""
    print()
    print("=" * 110)
    print("CHECKPOINT EVAL SUMMARY (RL data, 20 calendars)")
    print("=" * 110)

    cat_shorts = [CATEGORY_SHORT[c] for c in CATEGORY_ORDER]
    header = f"{'Ckpt':>6} {'Ep':>3} {'TrLoss':>7} {'EvLoss':>7} {'Overall':>12}"
    for s in cat_shorts:
        header += f" {s:>8}"
    print(header)
    print("-" * 110)

    for step, ckpt_path in checkpoints:
        epoch = get_checkpoint_epoch(ckpt_path)
        epoch_int = int(epoch) if epoch else "?"
        tl, el = losses.get(epoch_int, (None, None))
        tl_str = f"{tl:.4f}" if tl is not None else "-"
        el_str = f"{el:.4f}" if el is not None else "-"

        data = load_eval_result(step)
        if data:
            rl = data.get("rl", {})
            c, t = rl.get("correct", 0), rl.get("total", 0)
            pct = c / t * 100 if t > 0 else 0
            overall = f"{c}/{t} ({pct:.1f}%)"
            by_cat = rl.get("by_category", {})
            cat_strs = []
            for cat in CATEGORY_ORDER:
                cd = by_cat.get(cat, {})
                cc, ct = cd.get("correct", 0), cd.get("total", 0)
                cat_strs.append(f"{cc}/{ct}" if ct > 0 else "-")
        else:
            overall = "pending"
            cat_strs = ["-"] * len(CATEGORY_ORDER)

        line = f"{step:>6} {epoch_int:>3} {tl_str:>7} {el_str:>7} {overall:>12}"
        for cs in cat_strs:
            line += f" {cs:>8}"
        print(line)

    print()
    print(f"Per-checkpoint details: {EVAL_DIR}/checkpoint-*.json")
    print(f"Summary CSV:           {SUMMARY_CSV}")


# ── Main ──────────────────────────────────────────────────────

def main():
    os.makedirs(EVAL_DIR, exist_ok=True)

    checkpoints = discover_checkpoints()
    if not checkpoints:
        print(f"No checkpoints found in {SFT_OUTPUT}")
        return

    losses = read_epoch_losses()

    to_eval = [(s, p) for s, p in checkpoints if not is_evaluated(s)]
    n_done = len(checkpoints) - len(to_eval)

    print(f"Checkpoints: {len(checkpoints)} found, {n_done} evaluated, {len(to_eval)} pending")
    for step, path in checkpoints:
        status = "done" if is_evaluated(step) else "NEW"
        epoch = get_checkpoint_epoch(path)
        print(f"  checkpoint-{step} (epoch {int(epoch) if epoch else '?'}): {status}")

    if not to_eval:
        print("\nNothing new to evaluate.")
        print_summary(checkpoints, losses)
        return

    for step, ckpt_path in to_eval:
        epoch = get_checkpoint_epoch(ckpt_path)
        epoch_int = int(epoch) if epoch else "?"
        tl, el = losses.get(epoch_int, (None, None))

        print(f"\n{'='*60}")
        print(f"Checkpoint {step} (epoch {epoch_int})")
        if tl is not None:
            print(f"  Train loss: {tl:.4f}, Eval loss: {el:.4f}")
        print(f"{'='*60}")

        # Merge LoRA → fp16
        kill_vllm()
        if not merge_checkpoint(step):
            continue

        # Start vLLM
        vllm_proc = start_vllm()
        if vllm_proc is None:
            continue

        # Eval on RL data (with vLLM restart on hang)
        MAX_RETRIES = 3
        raw = None
        tmp_path = os.path.join(EVAL_DIR, f"_tmp_{step}.json")

        for attempt in range(1, MAX_RETRIES + 1):
            print(f"\n--- Evaluating on RL data (20 calendars) [attempt {attempt}/{MAX_RETRIES}] ---")
            raw = run_eval(tmp_path, num_calendars=20)

            n_results = len(raw["rl"].get("results", [])) if raw and "rl" in raw else 0
            n_expected = raw["rl"].get("total", 280) if raw and "rl" in raw else 280
            if n_results >= n_expected:
                break  # Full eval completed

            # Partial or failed — vLLM likely hung
            if attempt < MAX_RETRIES:
                n_done = n_results
                print(f"\n  Partial eval ({n_done}/280) — restarting vLLM for retry...")
                vllm_proc.kill()
                vllm_proc.wait()
                kill_vllm()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                vllm_proc = start_vllm()
                if vllm_proc is None:
                    raw = None
                    break
            else:
                print(f"\n  Eval incomplete after {MAX_RETRIES} attempts, using partial results")

        # Shut down vLLM
        vllm_proc.kill()
        vllm_proc.wait()
        kill_vllm()

        if not raw or "rl" not in raw:
            print(f"Eval FAILED for checkpoint-{step}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue

        # Enrich with metadata and category breakdown, save final
        rl = raw["rl"]
        rl["by_category"] = category_breakdown(rl.get("results", []))
        enriched = {
            "checkpoint": step,
            "epoch": epoch_int,
            "train_loss": tl,
            "eval_loss": el,
            "rl": rl,
        }
        result_path = os.path.join(EVAL_DIR, f"checkpoint-{step}.json")
        with open(result_path, "w") as f:
            json.dump(enriched, f, indent=2)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        c, t = rl["correct"], rl["total"]
        pct = c / t * 100 if t > 0 else 0
        print(f"\n>>> Checkpoint {step} (epoch {epoch_int}): {c}/{t} ({pct:.1f}%)")

        # Update summary after each checkpoint
        write_summary(checkpoints, losses)

    # Final summary
    print_summary(checkpoints, losses)


if __name__ == "__main__":
    main()
