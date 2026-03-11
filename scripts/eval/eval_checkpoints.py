#!/usr/bin/env python3
"""Evaluate each SFT checkpoint by merging LoRA, serving via vLLM, and running eval.

For each checkpoint:
1. Merge LoRA adapter into base model (fp16)
2. Start vLLM server with hermes tool-call parser
3. Run eval_qwen.py on calendars 0 and 5
4. Kill server, move to next checkpoint

Usage:
    python eval_checkpoints.py
"""

import json
import os
import signal
import subprocess
import sys
import time

PYTHON = os.environ.get("CONDA_PYTHON", "/home/abhor/miniconda3/envs/agentic/bin/python")
from calendar_agent.paths import PROJECT_ROOT, SFT_OUTPUT_DIR
PROJECT_DIR = str(PROJECT_ROOT)
SFT_OUTPUT = str(SFT_OUTPUT_DIR)
MERGED_DIR = os.path.join(SFT_OUTPUT, "merged_eval")
RESULTS_DIR = os.path.join(PROJECT_DIR, "checkpoint_eval_results")
VLLM_LOG = os.path.join(PROJECT_DIR, "vllm_eval.log")

CHECKPOINTS = ["checkpoint-36", "checkpoint-72", "checkpoint-108", "checkpoint-144", "checkpoint-180"]
EVAL_CALENDARS = [0, 5]
VLLM_PORT = 8000
MODEL_NAME = "sft-eval"

os.makedirs(RESULTS_DIR, exist_ok=True)


def merge_checkpoint(ckpt_path: str):
    """Merge LoRA checkpoint into fp16 model."""
    print(f"\n{'='*60}")
    print(f"Merging: {ckpt_path}")
    print(f"{'='*60}")

    merge_script = f"""
import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from unsloth import FastLanguageModel
import shutil

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
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        print(f"MERGE FAILED:\n{result.stderr[-500:]}")
        return False
    print("Merge OK.")
    return True


def start_vllm():
    """Start vLLM server and wait for it to be ready."""
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["PATH"] = f"/home/abhor_gupta/miniconda3/envs/agentic/bin:{env.get('PATH', '')}"

    with open(VLLM_LOG, "w") as logf:
        proc = subprocess.Popen(
            [
                PYTHON, "-m", "vllm.entrypoints.openai.api_server",
                "--model", MERGED_DIR,
                "--served-model-name", MODEL_NAME,
                "--enable-auto-tool-choice",
                "--tool-call-parser", "hermes",
                "--max-model-len", "3072",
                "--gpu-memory-utilization", "0.85",
                "--port", str(VLLM_PORT),
            ],
            stdout=logf, stderr=logf, env=env,
        )

    # Wait for server to be ready
    import urllib.request
    for attempt in range(60):
        time.sleep(5)
        try:
            resp = urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/v1/models", timeout=5)
            if resp.status == 200:
                print("vLLM server ready.")
                return proc
        except Exception:
            pass
    print("vLLM server failed to start!")
    proc.kill()
    return None


def kill_vllm(proc):
    """Kill vLLM server and all child processes."""
    if proc:
        proc.kill()
        proc.wait()
    # Kill any remaining GPU processes
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        for pid in result.stdout.strip().split("\n"):
            pid = pid.strip()
            if pid:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                except (ProcessLookupError, ValueError):
                    pass
    except Exception:
        pass
    time.sleep(3)


def run_eval(cal_idx: int, save_path: str):
    """Run eval_qwen.py on a calendar."""
    result = subprocess.run(
        [
            PYTHON, str(PROJECT_ROOT / "scripts" / "eval" / "eval_qwen.py"), str(cal_idx),
            "--model", MODEL_NAME,
            "--sft-data",
            "--with-final-answer",
            "--save", save_path,
        ],
        capture_output=True, text=True, timeout=600,
        cwd=PROJECT_DIR,
    )
    if result.returncode != 0:
        print(f"  Eval failed: {result.stderr[-300:]}")
        return None

    try:
        with open(save_path) as f:
            data = json.load(f)
        correct = sum(1 for t in data if t["eval_verdict"] == "Correct")
        total = len(data)
        return {"correct": correct, "total": total, "rate": correct / total if total > 0 else 0}
    except Exception as e:
        print(f"  Failed to parse results: {e}")
        return None


def main():
    all_results = {}

    for ckpt_name in CHECKPOINTS:
        ckpt_path = os.path.join(SFT_OUTPUT, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_name} (not found)")
            continue

        epoch = CHECKPOINTS.index(ckpt_name) + 1
        print(f"\n{'#'*60}")
        print(f"# Checkpoint: {ckpt_name} (Epoch {epoch})")
        print(f"{'#'*60}")

        # Merge
        if not merge_checkpoint(ckpt_path):
            all_results[ckpt_name] = {"epoch": epoch, "error": "merge_failed"}
            continue

        # Start vLLM
        proc = start_vllm()
        if not proc:
            all_results[ckpt_name] = {"epoch": epoch, "error": "vllm_failed"}
            continue

        # Evaluate
        ckpt_results = {"epoch": epoch, "calendars": {}}
        for cal_idx in EVAL_CALENDARS:
            save_path = os.path.join(RESULTS_DIR, f"{ckpt_name}_cal{cal_idx}.json")
            print(f"\n  Evaluating calendar {cal_idx}...")
            result = run_eval(cal_idx, save_path)
            if result:
                ckpt_results["calendars"][str(cal_idx)] = result
                print(f"  Cal {cal_idx}: {result['correct']}/{result['total']} ({result['rate']:.1%})")
            else:
                ckpt_results["calendars"][str(cal_idx)] = {"error": "eval_failed"}

        # Aggregate
        total_correct = sum(
            r.get("correct", 0) for r in ckpt_results["calendars"].values() if isinstance(r, dict) and "correct" in r
        )
        total_queries = sum(
            r.get("total", 0) for r in ckpt_results["calendars"].values() if isinstance(r, dict) and "total" in r
        )
        ckpt_results["total_correct"] = total_correct
        ckpt_results["total_queries"] = total_queries
        ckpt_results["total_rate"] = total_correct / total_queries if total_queries > 0 else 0

        all_results[ckpt_name] = ckpt_results

        # Kill vLLM
        kill_vllm(proc)

    # Summary
    print(f"\n{'='*60}")
    print("CHECKPOINT EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<20} {'Epoch':<8} {'Correct':<10} {'Total':<8} {'Rate':<8}")
    print("-" * 54)
    for ckpt_name, result in all_results.items():
        if "error" in result:
            print(f"{ckpt_name:<20} {result['epoch']:<8} {'ERROR: ' + result['error']}")
        else:
            print(
                f"{ckpt_name:<20} {result['epoch']:<8} "
                f"{result['total_correct']:<10} {result['total_queries']:<8} "
                f"{result['total_rate']:.1%}"
            )

    # Save summary
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
