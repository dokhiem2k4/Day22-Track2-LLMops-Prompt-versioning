"""
Run all 4 steps sequentially (or a specific step with --step N).

Usage:
    python run_all.py           # run all steps
    python run_all.py --step 3  # run only step 3
"""

import sys
import argparse
import importlib
import time


STEPS = [
    (1, "01_langsmith_rag_pipeline",  "Step 1: LangSmith RAG Pipeline"),
    (2, "02_prompt_hub_ab_routing",   "Step 2: Prompt Hub A/B Routing"),
    (3, "03_ragas_evaluation",        "Step 3: RAGAS Evaluation (~15-20 min)"),
    (4, "04_guardrails_validator",    "Step 4: Guardrails AI Validators"),
]


def run_step(module_name: str, label: str):
    print(f"\n{'#' * 60}")
    print(f"  Running: {label}")
    print(f"{'#' * 60}")
    t0 = time.time()
    mod = importlib.import_module(module_name)
    mod.main()
    elapsed = time.time() - t0
    print(f"\n  [{module_name}] done in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Run Day22 lab steps")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4],
                        help="Run only this step number")
    args = parser.parse_args()

    steps_to_run = [s for s in STEPS if args.step is None or s[0] == args.step]

    print("=" * 60)
    print("  Day 22 Lab — LangSmith + Prompt Versioning")
    print(f"  Running: {[s[2] for s in steps_to_run]}")
    print("=" * 60)

    for _, module_name, label in steps_to_run:
        run_step(module_name, label)

    print("\n" + "=" * 60)
    print("  All steps complete!")
    print("  Check evidence/ folder for screenshots and logs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
