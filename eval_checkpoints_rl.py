#!/usr/bin/env python3
"""Evaluate each SFT checkpoint on rl_data (held-out validation).

For each checkpoint: merge LoRA, start vLLM, eval on 5 rl_data calendars, kill server.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PYTHON = "/home/abhor_gupta/miniconda3/envs/agentic/bin/python"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
SFT_OUTPUT = os.path.join(PROJECT_DIR, "sft_output")
MERGED_DIR = os.path.join(SFT_OUTPUT, "merged_eval")
RESULTS_DIR = os.path.join(PROJECT_DIR, "checkpoint_eval_results", "rl_validation")
VLLM_LOG = os.path.join(PROJECT_DIR, "vllm_rl_eval.log")

CHECKPOINTS = ["checkpoint-36", "checkpoint-72", "checkpoint-108", "checkpoint-144", "checkpoint-180"]
EVAL_CALENDARS = [0, 5, 10, 15, 20]
VLLM_PORT = 8000
MODEL_NAME = "sft-eval"

os.makedirs(RESULTS_DIR, exist_ok=True)


def merge_checkpoint(ckpt_path):
    print(f"\nMerging: {ckpt_path}")
    merge_script = f"""
import os, shutil
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
from unsloth import FastLanguageModel
ckpt = "{ckpt_path}"
out = "{MERGED_DIR}"
if os.path.exists(out):
    shutil.rmtree(out)
model, tokenizer = FastLanguageModel.from_pretrained(model_name=ckpt, max_seq_length=4096, load_in_4bit=True)
model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
print("Merge complete.")
"""
    result = subprocess.run([PYTHON, "-c", merge_script], capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"MERGE FAILED:\n{result.stderr[-500:]}")
        return False
    print("Merge OK.")
    return True


def start_vllm():
    env = os.environ.copy()
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["PATH"] = f"/home/abhor_gupta/miniconda3/envs/agentic/bin:{env.get('PATH', '')}"
    with open(VLLM_LOG, "w") as logf:
        proc = subprocess.Popen(
            [PYTHON, "-m", "vllm.entrypoints.openai.api_server",
             "--model", MERGED_DIR, "--served-model-name", MODEL_NAME,
             "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
             "--max-model-len", "3072", "--gpu-memory-utilization", "0.85",
             "--port", str(VLLM_PORT)],
            stdout=logf, stderr=logf, env=env,
        )
    for attempt in range(60):
        time.sleep(5)
        try:
            resp = urllib.request.urlopen(f"http://localhost:{VLLM_PORT}/v1/models", timeout=5)
            if resp.status == 200:
                print("vLLM ready.")
                return proc
        except Exception:
            pass
    print("vLLM failed to start!")
    proc.kill()
    return None


def kill_vllm(proc):
    if proc:
        proc.kill()
        proc.wait()
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


def run_eval(cal_idx, save_path):
    result = subprocess.run(
        [PYTHON, "eval_qwen.py", str(cal_idx),
         "--model", MODEL_NAME, "--rl-data", "--with-final-answer", "--save", save_path],
        capture_output=True, text=True, timeout=600, cwd=PROJECT_DIR,
    )
    if result.returncode != 0:
        print(f"  Eval failed cal {cal_idx}: {result.stderr[-200:]}")
        return None
    try:
        with open(save_path) as f:
            data = json.load(f)
        correct = sum(1 for t in data if t["eval_verdict"] == "Correct")
        return {"correct": correct, "total": len(data), "rate": correct / len(data) if data else 0}
    except Exception as e:
        print(f"  Parse failed: {e}")
        return None


def main():
    all_results = {}
    for ckpt_name in CHECKPOINTS:
        ckpt_path = os.path.join(SFT_OUTPUT, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"Skipping {ckpt_name}")
            continue

        epoch = CHECKPOINTS.index(ckpt_name) + 1
        print(f"\n{'#'*60}")
        print(f"# {ckpt_name} (Epoch {epoch})")
        print(f"{'#'*60}")

        # Skip checkpoint-144 (already done)
        existing_results = {}
        skip = True
        for cal_idx in EVAL_CALENDARS:
            save_path = os.path.join(RESULTS_DIR, f"{ckpt_name}_cal{cal_idx}.json")
            if os.path.exists(save_path):
                try:
                    with open(save_path) as f:
                        data = json.load(f)
                    correct = sum(1 for t in data if t["eval_verdict"] == "Correct")
                    existing_results[str(cal_idx)] = {"correct": correct, "total": len(data), "rate": correct / len(data)}
                except Exception:
                    skip = False
                    break
            else:
                skip = False
                break

        if skip and len(existing_results) == len(EVAL_CALENDARS):
            print(f"  Using cached results for {ckpt_name}")
            ckpt_results = {"epoch": epoch, "calendars": existing_results}
        else:
            if not merge_checkpoint(ckpt_path):
                all_results[ckpt_name] = {"epoch": epoch, "error": "merge_failed"}
                continue
            proc = start_vllm()
            if not proc:
                all_results[ckpt_name] = {"epoch": epoch, "error": "vllm_failed"}
                continue

            ckpt_results = {"epoch": epoch, "calendars": {}}
            for cal_idx in EVAL_CALENDARS:
                save_path = os.path.join(RESULTS_DIR, f"{ckpt_name}_cal{cal_idx}.json")
                print(f"  Eval calendar {cal_idx}...")
                result = run_eval(cal_idx, save_path)
                if result:
                    ckpt_results["calendars"][str(cal_idx)] = result
                    print(f"  Cal {cal_idx}: {result['correct']}/{result['total']} ({result['rate']:.1%})")
                else:
                    ckpt_results["calendars"][str(cal_idx)] = {"error": "eval_failed"}
            kill_vllm(proc)

        total_correct = sum(r.get("correct", 0) for r in ckpt_results["calendars"].values() if "correct" in r)
        total_queries = sum(r.get("total", 0) for r in ckpt_results["calendars"].values() if "total" in r)
        ckpt_results["total_correct"] = total_correct
        ckpt_results["total_queries"] = total_queries
        ckpt_results["total_rate"] = total_correct / total_queries if total_queries > 0 else 0
        all_results[ckpt_name] = ckpt_results

    # Summary
    print(f"\n{'='*60}")
    print("RL_DATA VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<20} {'Epoch':<8} {'Correct':<10} {'Total':<8} {'Rate':<8}")
    print("-" * 54)
    for ckpt_name, result in all_results.items():
        if "error" in result:
            print(f"{ckpt_name:<20} {result['epoch']:<8} ERROR: {result['error']}")
        else:
            print(f"{ckpt_name:<20} {result['epoch']:<8} {result['total_correct']:<10} {result['total_queries']:<8} {result['total_rate']:.1%}")

    summary_path = os.path.join(RESULTS_DIR, "summary_all.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
