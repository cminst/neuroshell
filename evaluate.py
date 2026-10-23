"""Evaluate a saved K+2 NeuroShell policy on generated held-out arrays."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from neuroshell import (
    ActorCritic,
    ALL_FAMILY_NAMES,
    BASELINES,
    FAMILIES,
    PAPER_FAMILY_NAMES,
    log_strides,
    make_candidate_gaps,
    run_policy,
    run_shellsort,
)


def mean(values: list[float]) -> float:
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes-per-family", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=223000, help="held-out array seed")
    parser.add_argument(
        "--family-set",
        choices=("auto", "seven", "thirteen"),
        default="auto",
        help="evaluation families; auto uses the mixture recorded in the checkpoint",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episodes_per_family < 1:
        raise ValueError("--episodes-per-family must be positive")

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint["config"]
    recorded_families = tuple(config.get("family_names", PAPER_FAMILY_NAMES))
    family_names = {
        "auto": recorded_families,
        "seven": PAPER_FAMILY_NAMES,
        "thirteen": ALL_FAMILY_NAMES,
    }[args.family_set]
    network = ActorCritic(
        observation_dim=len(log_strides(config["n"])) + 2,
        action_dim=len(make_candidate_gaps(config["n"])),
        hidden=config["hidden"],
    ).to(args.device)
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    print(f"evaluation families ({len(family_names)}): {', '.join(family_names)}")
    columns = ["Policy", *BASELINES]
    totals = {column: [] for column in columns}
    print("| Family | " + " | ".join(columns) + " | Sorted | Policy vs. Sedgewick |")
    print("|---|" + "---:|" * (len(columns) + 2))
    for family_index, family in enumerate(family_names):
        arrays = [
            FAMILIES[family](config["n"], np.random.default_rng(args.seed + family_index * 100_000 + episode * 1000))
            for episode in range(args.episodes_per_family)
        ]
        policy_runs = [run_policy(network, config["n"], config["max_passes"], array, args.device) for array in arrays]
        policy_counts = [run["comparisons"] for run in policy_runs]
        values = {"Policy": mean(policy_counts)}
        for name, gaps in BASELINES.items():
            values[name] = mean([run_shellsort(array, gaps(config["n"]))[0] for array in arrays])
        for name, value in values.items():
            totals[name].append(value)
        improvement = (values["Policy"] / values["Sedgewick"] - 1.0) * 100.0
        sorted_fraction = mean([float(run["sorted"]) for run in policy_runs])
        print("| " + family + " | " + " | ".join(f"{values[column]:.1f}" for column in columns) + f" | {sorted_fraction:.1%} | {improvement:+.1f}% |")

    averages = {name: mean(values) for name, values in totals.items()}
    improvement = (averages["Policy"] / averages["Sedgewick"] - 1.0) * 100.0
    print("| **Average** | " + " | ".join(f"**{averages[column]:.1f}**" for column in columns) + f" | — | **{improvement:+.1f}%** |")


if __name__ == "__main__":
    main()
