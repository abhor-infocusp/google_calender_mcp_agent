#!/usr/bin/env python3
"""Evaluate SFT checkpoints on RL data: peft-merge LoRA, serve via vLLM on MIG slice 2, run eval.

Discovers checkpoints dynamically from $RUN_DIR/checkpoints/checkpoint-*.
Skips already-evaluated ones (checks $RUN_DIR/eval/checkpoint-{N}.json).
Skips already-merged ones (checks $RUN_DIR/eval/merged_tmp_{N}/config.json).

Results:
    $RUN_DIR/eval/checkpoint-{N}.json   per-checkpoint detailed results
    $RUN_DIR/eval/summary.csv           one row per checkpoint

Usage:
    PYTHONPATH=src python scripts/eval/eval_all_checkpoints.py
    RUN_DIR=runs/sft_v6_qwen3_14b_20260420 PYTHONPATH=src python scripts/eval/eval_all_checkpoints.py
"""

import csv
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from calendar_agent.paths import PROJECT_ROOT

PYTHON = sys.executable
PROJECT = str(PROJECT_ROOT)
RUN_DIR = os.environ.get("RUN_DIR", "runs/sft_v6_qwen3_14b_20260420")
CKPT_DIR = os.path.join(RUN_DIR, "checkpoints")
EVAL_SUBDIR = os.environ.get("EVAL_SUBDIR", "eval")
EVAL_DIR = os.path.join(RUN_DIR, EVAL_SUBDIR)
LOG_DIR = os.path.join(EVAL_DIR, "logs")
SUMMARY_CSV = os.path.join(EVAL_DIR, "summary.csv")
LOSS_CSV = os.path.join(RUN_DIR, "diagnostics", "epoch_losses.csv")

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen3-14B")
MIG_UUID = os.environ.get("MIG_UUID", "MIG-dd607cdf-e8cb-531f-b478-417160625a35")
PORT = int(os.environ.get("EVAL_PORT", "8006"))
SERVED_NAME = os.environ.get("SERVED_NAME", "ckpt-eval")
EVAL_MODE = os.environ.get("EVAL_MODE", "rl")  # "rl" | "test"
NUM_CALENDARS = int(os.environ.get("NUM_CALENDARS", "20"))

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
    checkpoints = []
    if not os.path.isdir(CKPT_DIR):
        return checkpoints
    for name in os.listdir(CKPT_DIR):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m:
            step = int(m.group(1))
            checkpoints.append((step, os.path.join(CKPT_DIR, name)))
    return sorted(checkpoints)


def get_checkpoint_epoch(ckpt_path):
    state_path = os.path.join(ckpt_path, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            return json.load(f).get("epoch", None)
    return None


def read_epoch_losses():
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
                    losses[int(parts[0])] = (float(parts[1]), float(parts[2]))
                except ValueError:
                    continue
    return losses


def is_evaluated(step):
    return os.path.exists(os.path.join(EVAL_DIR, f"checkpoint-{step}.json"))


def load_eval_result(step):
    path = os.path.join(EVAL_DIR, f"checkpoint-{step}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def category_breakdown(results):
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


# ── Merge & Serve ─────────────────────────────────────────────

def merged_dir(step):
    return os.path.join(EVAL_DIR, f"merged_tmp_{step}")


def merge_checkpoint(step, ckpt_path):
    """Merge LoRA via peft on CPU, save bf16 fp16 shards."""
    out = merged_dir(step)
    if os.path.exists(os.path.join(out, "config.json")):
        print(f"  merged_tmp_{step} already exists — skipping merge")
        return True

    print(f"  Merging checkpoint-{step} via peft (CPU, bf16)...")
    script = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    {BASE_MODEL!r}, torch_dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True,
)
tok = AutoTokenizer.from_pretrained({ckpt_path!r})
model = PeftModel.from_pretrained(model, {ckpt_path!r}, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()
model.save_pretrained({out!r}, safe_serialization=True, max_shard_size="5GB")
tok.save_pretrained({out!r})
print("Merge complete")
"""
    script_path = os.path.join(EVAL_DIR, f"_merge_{step}.py")
    with open(script_path, "w") as f:
        f.write(script)

    log_path = os.path.join(LOG_DIR, f"merge_{step}.log")
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            [PYTHON, script_path],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONUNBUFFERED": "1"},
            stdout=log_f, stderr=subprocess.STDOUT,
            timeout=600,
        )
    os.remove(script_path)

    if result.returncode != 0 or not os.path.exists(os.path.join(out, "config.json")):
        print(f"  MERGE FAILED (exit={result.returncode}) — see {log_path}")
        return False
    print(f"  Merge OK -> {out}")
    return True


def kill_vllm_on_port():
    """SIGKILL anything bound to our port AND any lingering VLLM::EngineCore
    subprocs. Then wait long enough for CUDA to actually release the memory
    — the vLLM engine core runs as a subprocess and a short sleep isn't
    enough on a 14B model (seen 23 GiB stranded when sleep was 3s)."""
    try:
        r = subprocess.run(f"lsof -ti :{PORT}", shell=True, capture_output=True, text=True)
        pids = [p for p in r.stdout.strip().split("\n") if p]
        # Also kill any orphaned VLLM::EngineCore owned by this user — they
        # are separate subprocesses that outlive their parent's SIGKILL.
        r2 = subprocess.run("pgrep -u $USER -f 'VLLM::EngineCore' | head -20",
                            shell=True, capture_output=True, text=True)
        engine_pids = [p for p in r2.stdout.strip().split("\n") if p]
        for pid in pids + engine_pids:
            try:
                os.kill(int(pid), signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass
    except Exception:
        pass
    # Poll for processes to exit + CUDA to flush. 60s should be plenty.
    for _ in range(60):
        time.sleep(1)
        try:
            remaining = subprocess.run(f"lsof -ti :{PORT}", shell=True,
                                        capture_output=True, text=True).stdout.strip()
            if not remaining:
                break
        except Exception:
            pass
    # Belt-and-suspenders: additional wait for CUDA allocator to fully drop.
    time.sleep(30)


def start_vllm(step):
    """Launch vLLM pinned to MIG slice 2 with fp8 quant, serving merged model."""
    out = merged_dir(step)
    launcher = os.path.join(EVAL_DIR, f"_serve_{step}.py")
    launcher_src = f"""import os, sys


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = {MIG_UUID!r}
    sys.argv = [
        "vllm",
        "--model", {out!r},
        "--served-model-name", {SERVED_NAME!r},
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--enforce-eager",
        "--quantization", "fp8",
        "--port", "{PORT}",
    ]
    import runpy
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")


if __name__ == "__main__":
    main()
"""
    with open(launcher, "w") as f:
        f.write(launcher_src)

    vllm_log = os.path.join(LOG_DIR, f"vllm_{step}.log")
    print(f"  Starting vLLM on port {PORT} (slice 2, fp8)... log: {vllm_log}")
    log_f = open(vllm_log, "w")
    proc = subprocess.Popen(
        [PYTHON, launcher],
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        stdout=log_f, stderr=subprocess.STDOUT,
    )

    for _ in range(180):  # up to 9 minutes
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://localhost:{PORT}/v1/models", timeout=2)
            if r.status == 200:
                print(f"  vLLM ready on port {PORT}")
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            log_f.close()
            tail = open(vllm_log).read()[-1200:]
            print(f"  vLLM exited early:\n{tail}")
            return None

    print(f"  vLLM failed to start within timeout")
    proc.kill()
    return None


def run_eval(save_path, num_calendars=None):
    if num_calendars is None:
        num_calendars = NUM_CALENDARS
    cmd = [
        PYTHON, "-u", os.path.join(PROJECT, "scripts/eval/eval_batch.py"),
        "--mode", EVAL_MODE,
        "--model", SERVED_NAME,
        "--base-url", f"http://localhost:{PORT}/v1",
        "--num-calendars", str(num_calendars),
        "--max-queries", "0",
        "--save", save_path,
    ]
    subprocess.run(
        cmd,
        env={**os.environ, "PYTHONPATH": os.path.join(PROJECT, "src"), "PYTHONUNBUFFERED": "1"},
        timeout=14400,
    )
    if os.path.exists(save_path):
        with open(save_path) as f:
            return json.load(f)
    return None


# ── Results I/O ───────────────────────────────────────────────

def write_summary(checkpoints, losses):
    header = ["checkpoint", "epoch", "train_loss", "eval_loss", "correct", "total", "pct"]
    for cat in CATEGORY_ORDER:
        header.append(CATEGORY_SHORT[cat])

    rows = []
    for step, _ in checkpoints:
        data = load_eval_result(step)
        if not data:
            continue
        rl = data.get(EVAL_MODE, {})
        correct, total = rl.get("correct", 0), rl.get("total", 0)
        pct = round(correct / total * 100, 1) if total > 0 else 0
        by_cat = rl.get("by_category", {})
        row = [step, data.get("epoch", ""), data.get("train_loss", ""),
               data.get("eval_loss", ""), correct, total, pct]
        for cat in CATEGORY_ORDER:
            cd = by_cat.get(cat, {})
            c, t = cd.get("correct", 0), cd.get("total", 0)
            row.append(f"{c}/{t}" if t > 0 else "")
        rows.append(row)

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def print_summary(checkpoints, losses):
    print()
    print("=" * 110)
    print(f"CHECKPOINT EVAL SUMMARY ({EVAL_MODE} data, {NUM_CALENDARS} calendars) — {RUN_DIR}")
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
        tl, el = losses.get(epoch_int if isinstance(epoch_int, int) else -1, (None, None))
        tl_str = f"{tl:.4f}" if tl is not None else "-"
        el_str = f"{el:.4f}" if el is not None else "-"

        data = load_eval_result(step)
        if data:
            rl = data.get(EVAL_MODE, {})
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
    os.makedirs(LOG_DIR, exist_ok=True)

    only = os.environ.get("ONLY_CHECKPOINT")  # e.g. "4659" for single-ckpt runs

    checkpoints = discover_checkpoints()
    if not checkpoints:
        print(f"No checkpoints found in {CKPT_DIR}")
        return

    losses = read_epoch_losses()

    to_eval = [(s, p) for s, p in checkpoints if not is_evaluated(s)]
    if only:
        to_eval = [(s, p) for s, p in to_eval if str(s) == only]
    n_done = len(checkpoints) - len([s for s, _ in checkpoints if not is_evaluated(s)])

    print(f"Checkpoints: {len(checkpoints)} found, {n_done} evaluated, {len(to_eval)} to run")
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
        tl, el = losses.get(epoch_int if isinstance(epoch_int, int) else -1, (None, None))

        print(f"\n{'='*60}")
        print(f"Checkpoint {step} (epoch {epoch_int})")
        if tl is not None:
            print(f"  Train loss: {tl:.4f}, Eval loss: {el:.4f}")
        print(f"{'='*60}")

        kill_vllm_on_port()
        if not merge_checkpoint(step, ckpt_path):
            continue

        vllm_proc = start_vllm(step)
        if vllm_proc is None:
            continue

        tmp_path = os.path.join(EVAL_DIR, f"_tmp_checkpoint-{step}.json")
        print(f"\n  Evaluating on {EVAL_MODE} data ({NUM_CALENDARS} calendars)...")
        raw = run_eval(tmp_path)

        try:
            vllm_proc.kill()
            vllm_proc.wait(timeout=15)
        except Exception:
            pass
        kill_vllm_on_port()

        if not raw or EVAL_MODE not in raw:
            print(f"  Eval FAILED for checkpoint-{step}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue

        rl = raw[EVAL_MODE]
        rl["by_category"] = category_breakdown(rl.get("results", []))
        enriched = {
            "checkpoint": step,
            "epoch": epoch_int,
            "train_loss": tl,
            "eval_loss": el,
            EVAL_MODE: rl,
        }
        result_path = os.path.join(EVAL_DIR, f"checkpoint-{step}.json")
        with open(result_path, "w") as f:
            json.dump(enriched, f, indent=2)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        c, t = rl["correct"], rl["total"]
        pct = c / t * 100 if t > 0 else 0
        print(f"\n>>> Checkpoint {step} (epoch {epoch_int}): {c}/{t} ({pct:.1f}%)")

        write_summary(checkpoints, losses)

    print_summary(checkpoints, losses)


if __name__ == "__main__":
    main()
