"""
Main Experiment Runner
======================
Orchestrates all GhostPermit experiments.

Usage:
    # Run single model, single combo
    python evaluation/run_experiment.py --experiment ghostpermit \
        --model llama-3.3-70b --combo C10 --trials 30

    # Run all models (sequential, ~8 hours)
    python evaluation/run_experiment.py --experiment ghostpermit --all-models

    # Run AgentWorm
    python evaluation/run_experiment.py --experiment agentworm

    # Run CoT hijacking
    python evaluation/run_experiment.py --experiment cot
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attack.ghostpermit_attack import run_attack, PERM_COMBOS, MODEL_CONFIGS
from attack.agentworm import run_worm, TOOL_STORES
from attack.cot_hijacking import run_cot_hijacking

ALL_MODELS = list(MODEL_CONFIGS.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["ghostpermit", "agentworm", "cot"],
                        required=True)
    parser.add_argument("--model",     default="llama-3.3-70b")
    parser.add_argument("--combo",     default="C10")
    parser.add_argument("--trials",    default=30, type=int)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--output",    default="data/")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.experiment == "ghostpermit":
        models = ALL_MODELS if args.all_models else [args.model]
        for model in models:
            print(f"\n{'='*60}\nGhostPermit: {model}\n{'='*60}")
            for combo in PERM_COMBOS:
                run_attack(model, combo, args.trials,
                           os.path.join(args.output, "ghostpermit_results.csv"))

    elif args.experiment == "agentworm":
        models = ALL_MODELS[:6] if args.all_models else [args.model]
        for model in models:
            for store in TOOL_STORES:
                print(f"\nAgentWorm: {model} | store={store}")
                run_worm(model, store, n_hops=5, n_trials=3,
                         output_csv=os.path.join(args.output,
                                                 "agentworm_results.csv"))

    elif args.experiment == "cot":
        models = ["llama-4-scout-17b", "llama-3.3-70b", "gpt-4o"]
        for model in models:
            print(f"\nCoT hijacking: {model}")
            run_cot_hijacking(model, n_trials=10,
                              output_csv=os.path.join(args.output,
                                                      "cot_hijacking_results.csv"))


if __name__ == "__main__":
    main()
