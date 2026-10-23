"""Compile one adaptive policy into one fixed Shellsort schedule per family."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from neuroshell import (
    ALL_FAMILY_NAMES,
    ActorCritic,
    BASELINES,
    FAMILIES,
    PAPER_FAMILY_NAMES,
    log_strides,
    make_candidate_gaps,
    run_policy,
    run_shellsort,
)


def generated_arrays(n: int, family: str, family_index: int, episodes: int, seed: int) -> list[np.ndarray]:
    return [
        FAMILIES[family](n, np.random.default_rng(seed + family_index * 100_000 + episode * 1000))
        for episode in range(episodes)
    ]


def score_schedule(schedule: tuple[int, ...], arrays: list[np.ndarray]) -> tuple[float, float]:
    counts, sorted_flags = [], []
    for array in arrays:
        comparisons, sorted_ok = run_shellsort(array, list(schedule))
        # Incomplete schedules are never preferred during validation.
        counts.append(comparisons if sorted_ok else comparisons + 100_000)
        sorted_flags.append(float(sorted_ok))
    return float(np.mean(counts)), float(np.mean(sorted_flags))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--family-set", choices=("auto", "seven", "thirteen"), default="auto")
    parser.add_argument("--compile-episodes", type=int, default=200)
    parser.add_argument("--validation-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=1000)
    parser.add_argument("--compile-seed", type=int, default=771000)
    parser.add_argument("--validation-seed", type=int, default=881000)
    parser.add_argument("--eval-seed", type=int, default=991000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=Path("compiled_schedules.json"))
    args = parser.parse_args()
    if min(args.compile_episodes, args.validation_episodes, args.eval_episodes) < 1:
        raise ValueError("all episode counts must be positive")

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    recorded_families = tuple(config.get("family_names", PAPER_FAMILY_NAMES))
    family_names = {
        "auto": recorded_families,
        "seven": PAPER_FAMILY_NAMES,
        "thirteen": ALL_FAMILY_NAMES,
    }[args.family_set]
    n, max_passes = config["n"], config["max_passes"]
    network = ActorCritic(
        observation_dim=len(log_strides(n)) + 2,
        action_dim=len(make_candidate_gaps(n)),
        hidden=config["hidden"],
    ).to(args.device)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    compiled: dict[str, dict] = {}
    print(f"compiling {len(family_names)} families: {', '.join(family_names)}")
    for family_index, family in enumerate(family_names):
        compile_arrays = generated_arrays(n, family, family_index, args.compile_episodes, args.compile_seed)
        validation_arrays = generated_arrays(n, family, family_index, args.validation_episodes, args.validation_seed)
        evaluation_arrays = generated_arrays(n, family, family_index, args.eval_episodes, args.eval_seed)

        candidates = Counter(
            tuple(run["gaps"])
            for run in (run_policy(network, n, max_passes, array, args.device) for array in compile_arrays)
            if run["sorted"]
        )
        if not candidates:
            candidates[(1,)] = 1

        validation = []
        for schedule, frequency in candidates.items():
            comparisons, sorted_rate = score_schedule(schedule, validation_arrays)
            validation.append(
                {
                    "schedule": list(schedule),
                    "compile_frequency": frequency,
                    "comparisons": comparisons,
                    "sorted_rate": sorted_rate,
                }
            )
        validation.sort(key=lambda row: row["comparisons"])
        selected = tuple(validation[0]["schedule"])

        compiled_comparisons, compiled_sorted_rate = score_schedule(selected, evaluation_arrays)
        policy_runs = [run_policy(network, n, max_passes, array, args.device) for array in evaluation_arrays]
        policy_comparisons = float(np.mean([run["comparisons"] for run in policy_runs]))
        baseline_comparisons = {
            name: float(np.mean([run_shellsort(array, gap_function(n))[0] for array in evaluation_arrays]))
            for name, gap_function in BASELINES.items()
        }
        result = {
            "selected_schedule": list(selected),
            "candidate_schedules": len(candidates),
            "compile_frequency": candidates[selected],
            "validation": validation,
            "compiled_eval_comparisons": compiled_comparisons,
            "compiled_eval_sorted_rate": compiled_sorted_rate,
            "live_policy_eval_comparisons": policy_comparisons,
            "live_policy_eval_sorted_rate": float(np.mean([run["sorted"] for run in policy_runs])),
            "baseline_eval_comparisons": baseline_comparisons,
            "compiled_vs_sedgewick_pct": (compiled_comparisons / baseline_comparisons["Sedgewick"] - 1.0) * 100.0,
        }
        compiled[family] = result
        print(
            f"{family:18s} {list(selected)}  "
            f"{compiled_comparisons:.1f} comparisons  "
            f"{result['compiled_vs_sedgewick_pct']:+.1f}% vs. Sedgewick"
        )

    output = {
        "checkpoint": str(args.checkpoint),
        "state_variant": checkpoint.get("state_variant"),
        "n": n,
        "max_passes": max_passes,
        "families": list(family_names),
        "compile_episodes": args.compile_episodes,
        "validation_episodes": args.validation_episodes,
        "eval_episodes": args.eval_episodes,
        "compile_seed": args.compile_seed,
        "validation_seed": args.validation_seed,
        "eval_seed": args.eval_seed,
        "compiled": compiled,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved compiled schedules: {args.output}")


if __name__ == "__main__":
    main()
