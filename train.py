"""Train one K+2 NeuroShell policy and save a checkpoint."""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from neuroshell import ALL_FAMILY_NAMES, PAPER_FAMILY_NAMES, PPOConfig, PPOTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=256, help="array length")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--max-passes", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument(
        "--family-set",
        choices=("seven", "thirteen"),
        default="seven",
        help="seven paper families or the expanded thirteen-family mixture",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()
    if args.n < 4 or args.updates < 1:
        raise ValueError("--n must be at least 4 and --updates must be positive")

    # Match the seed convention used for the paper's K+2 training runs.
    random.seed(args.seed + args.n)
    np.random.seed(args.seed + args.n)
    torch.manual_seed(args.seed + args.n)
    family_names = PAPER_FAMILY_NAMES if args.family_set == "seven" else ALL_FAMILY_NAMES
    config = PPOConfig(
        n=args.n,
        max_passes=args.max_passes,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        hidden=args.hidden,
        family_names=family_names,
    )
    trainer = PPOTrainer(
        config,
        total_updates=args.updates,
        seed=args.seed + 17 * args.n,
        device=args.device,
    )
    start = time.perf_counter()
    print(f"training families ({len(family_names)}): {', '.join(family_names)}")
    for update in range(1, args.updates + 1):
        metrics = trainer.train_step(update)
        if update == 1 or update % args.log_every == 0 or update == args.updates:
            print(
                f"update {update:4d}/{args.updates}  "
                f"policy_loss={metrics['policy_loss']:.4f}  "
                f"value_loss={metrics['value_loss']:.4f}  "
                f"elapsed={time.perf_counter() - start:.1f}s",
                flush=True,
            )

    output = args.output or Path("checkpoints") / f"policy_n{args.n}_seed{args.seed}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": trainer.network.state_dict(),
            "config": asdict(config),
            "seed": args.seed,
            "updates": args.updates,
            "state_variant": "k_plus_2_log_stride_inversions_only",
        },
        output,
    )
    print(f"saved checkpoint: {output}")


if __name__ == "__main__":
    main()
