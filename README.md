<h1 align="center">Learning Spectrum-Matched Gap Sequences for Shellsort</h1>

This repository is the official implementation of the paper **Learning Spectrum-Matched Gap Sequences for Shellsort** (NeurIPS 2026).

## Method

For an array of length `N`, we train an RL policy taking in an observation of size `K + 2` entries, where
`K = floor(log2(N))`:

- `K` inversion rates at strides `1, 2, 4, ..., N/2`;
- remaining-pass budget; and
- the preceding gap, normalized by `N`.

The action space is a bounded geometric set of candidate next gaps.  The policy is
a two-hidden-layer MLP trained with PPO and evaluated greedily. Refer to [the paper](https://openreview.net/forum?id=Ym8pFbqtzJ) for more details.

## Install

Python 3.12 was used for the reported experiments.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For a CPU-only PyTorch installation, follow the platform-specific command at
[pytorch.org](https://pytorch.org/get-started/locally/) before installing the
remaining requirements.

## Train

```bash
python train.py --n 256 --seed 20260809 --updates 500
```

The command writes a self-contained checkpoint under `checkpoints/`.  The
defaults match the paper training configuration: 32 parallel environments,
64 rollout steps, 500 PPO updates, and a maximum of 12 gap passes.

To train on the expanded mixture from the held-out-family experiment:

```bash
python train.py --n 256 --seed 20260809 --updates 500 --family-set thirteen
```

## Evaluate

```bash
python evaluate.py --checkpoint checkpoints/policy_n256_seed20260809.pt \
  --episodes-per-family 1000
```

Evaluation generates held-out arrays from the requested generators and
reports comparison counts for the learned policy and all seven classical
baselines: Ciura, Tokuda, Sedgewick, Knuth, Hibbard, Shell's original
halving sequence, and Pratt. Both scripts accept `--help` for all options.

A checkpoint trained with `--family-set thirteen` evaluates on all 13
families by default.  To explicitly select an evaluation mixture, pass
`--family-set seven` or `--family-set thirteen` to `evaluate.py`.

## Compile fixed schedules

To turn an adaptive policy into one fixed gap schedule per family:

```bash
python compile.py --checkpoint checkpoints/policy_n256_seed20260809.pt
```

For each family, compilation collects the policy's greedy schedules on a
compile-only split, chooses the best unique schedule on a separate validation
split, and evaluates that selected schedule on a third held-out split. The
resulting JSON includes the selected schedules, all validation candidates,
live-policy and compiled comparison counts, and all seven baselines.

## Reproducibility

Training and evaluation seeds are explicit command-line arguments.  The
array generators, candidate-gap construction, PPO hyperparameters, and
checkpoint metadata are all included in this repository.  Paper aggregates
use independent training seeds; a single invocation is intended as a compact
train-and-evaluate example rather than a replacement for that full sweep.

## Layout

```
neuroshell.py  # environment, synthetic generators, PPO, and baselines
train.py       # train and save one policy
evaluate.py    # evaluate one saved policy
compile.py     # compile one saved policy into fixed family schedules
```

## License

Released under the [MIT License](LICENSE).

