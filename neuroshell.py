"""K+2 NeuroShell: adaptive Shellsort gap selection with PPO."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


try:
    from numba import njit

    @njit(cache=True)
    def _shellsort_pass(array: np.ndarray, gap: int) -> int:
        comparisons = 0
        for index in range(gap, len(array)):
            value = array[index]
            cursor = index
            while cursor >= gap:
                comparisons += 1
                if array[cursor - gap] > value:
                    array[cursor] = array[cursor - gap]
                    cursor -= gap
                else:
                    break
            array[cursor] = value
        return comparisons

    def shellsort_pass(array: np.ndarray, gap: int) -> int:
        """Run one Shellsort insertion pass and return its comparison count."""
        return int(_shellsort_pass(array, gap))

except ImportError:

    def shellsort_pass(array: np.ndarray, gap: int) -> int:
        """Run one Shellsort insertion pass and return its comparison count."""
        comparisons = 0
        for index in range(gap, len(array)):
            value = array[index]
            cursor = index
            while cursor >= gap:
                comparisons += 1
                if array[cursor - gap] > value:
                    array[cursor] = array[cursor - gap]
                    cursor -= gap
                else:
                    break
            array[cursor] = value
        return comparisons


def run_shellsort(array: np.ndarray, gaps: list[int]) -> tuple[int, bool]:
    values = array.astype(np.float64, copy=True)
    comparisons = sum(shellsort_pass(values, gap) for gap in gaps)
    return comparisons, bool(np.all(values[:-1] <= values[1:]))


def make_random(n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.arange(n, dtype=np.float64)
    rng.shuffle(values)
    return values


def make_nearly_sorted(n: int, rng: np.random.Generator, frac: float = 0.05) -> np.ndarray:
    values = np.arange(n, dtype=np.float64)
    count = max(2, int(frac * n))
    indices = rng.choice(n, count, replace=False)
    shuffled = values[indices].copy()
    rng.shuffle(shuffled)
    values[indices] = shuffled
    return values


def make_reversed(n: int, rng: np.random.Generator, swap_frac: float = 0.02) -> np.ndarray:
    values = np.arange(n - 1, -1, -1, dtype=np.float64)
    for _ in range(max(1, int(swap_frac * n))):
        first = int(rng.integers(0, n))
        second = int(np.clip(first + rng.integers(-8, 9), 0, n - 1))
        values[first], values[second] = values[second], values[first]
    return values


def make_interleaved(n: int, rng: np.random.Generator) -> np.ndarray:
    count = int(rng.choice([3, 4, 5, 6, 8]))
    blocks = np.array_split(np.arange(n, dtype=np.float64), count)
    order = rng.permutation(count)
    return np.asarray(
        [blocks[block][index] for index in range(max(map(len, blocks)))
         for block in order if index < len(blocks[block])],
        dtype=np.float64,
    )


def make_pipe_organ(n: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.permutation(n).astype(np.float64)
    peak = int(rng.integers(max(1, int(0.35 * n)), min(n - 1, int(0.65 * n)) + 1))
    return np.concatenate([np.sort(values[:peak]), np.sort(values[peak:])[::-1]])


def make_sawtooth(n: int, rng: np.random.Generator) -> np.ndarray:
    count = int(rng.choice([4, 6, 8, 10]))
    blocks = [np.sort(block) for block in np.array_split(rng.permutation(n), count)]
    return np.asarray(
        [block[index] for index in range(max(map(len, blocks)))
         for block in blocks if index < len(block)],
        dtype=np.float64,
    )


def make_many_duplicates(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, max(2, int(0.1 * n)), size=n).astype(np.float64)


def make_rotated(n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.arange(n, dtype=np.float64)
    cut = int(rng.integers(n // 4, 3 * n // 4))
    return np.concatenate([values[cut:], values[:cut]])


def make_organ_reversed(n: int, rng: np.random.Generator) -> np.ndarray:
    values = rng.permutation(n).astype(np.float64)
    midpoint = n // 2
    return np.concatenate([np.sort(values[:midpoint])[::-1], np.sort(values[midpoint:])])


def make_sqrt_sorted(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sorted contiguous blocks of length about sqrt(n), in random block order."""
    num_blocks = max(1, int(np.sqrt(n)))
    blocks = list(np.array_split(np.arange(n, dtype=np.float64), num_blocks))
    rng.shuffle(blocks)
    return np.concatenate(blocks)


def make_anti_shellsort(n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.arange(n, dtype=np.float64)
    for gap in reversed(sedgewick_gaps(n)):
        for index in range(gap, n):
            if rng.random() < 0.5:
                values[index], values[index - gap] = values[index - gap], values[index]
    return values


def make_mostly_equal(n: int, rng: np.random.Generator) -> np.ndarray:
    values = np.full(n, n // 2, dtype=np.float64)
    count = max(1, n // 10)
    indices = rng.choice(n, count, replace=False)
    values[indices] = rng.integers(0, n, size=count)
    return values


def make_zipf(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.zipf(1.5, size=n).astype(np.float64)


FAMILIES: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "random": make_random,
    "nearly_sorted": make_nearly_sorted,
    "reversed": make_reversed,
    "interleaved": make_interleaved,
    "pipe_organ": make_pipe_organ,
    "sawtooth": make_sawtooth,
    "many_duplicates": make_many_duplicates,
    "rotated": make_rotated,
    "organ_reversed": make_organ_reversed,
    "sqrt_sorted": make_sqrt_sorted,
    "anti_shellsort": make_anti_shellsort,
    "mostly_equal": make_mostly_equal,
    "zipf": make_zipf,
}
PAPER_FAMILY_NAMES = tuple(list(FAMILIES)[:7])
HELD_OUT_FAMILY_NAMES = tuple(list(FAMILIES)[7:])
ALL_FAMILY_NAMES = tuple(FAMILIES)


def log_strides(n: int) -> list[int]:
    strides, stride = [], 1
    while stride <= n // 2:
        strides.append(stride)
        stride *= 2
    return strides


def inversion_spectrum(array: np.ndarray, strides: list[int]) -> np.ndarray:
    """The K array-dependent coordinates of the K+2 observation."""
    n = len(array)
    return np.asarray(
        [float(np.mean(array[: n - stride] > array[stride:])) for stride in strides],
        dtype=np.float32,
    )


def make_candidate_gaps(n: int) -> list[int]:
    """Bounded geometric candidate set used in the paper experiments."""
    maximum = n // 2
    gaps = set(range(1, min(17, maximum + 1)))
    gap = 16.0
    while gap <= maximum:
        gaps.add(int(round(gap)))
        gap *= 1.15
    for base in (2, 3):
        gap = 1
        while gap <= maximum:
            gaps.add(gap)
            gap *= base
    return sorted(gap for gap in gaps if 1 <= gap <= maximum)


def sedgewick_gaps(n: int) -> list[int]:
    gaps, index = set(), 0
    while True:
        gap = (9 * (2**index - 2 ** (index // 2)) + 1 if index % 2 == 0
               else 8 * 2**index - 6 * 2 ** ((index + 1) // 2) + 1)
        if gap >= n:
            break
        gaps.add(gap)
        index += 1
    return sorted(gaps | {1}, reverse=True)


def ciura_gaps(n: int) -> list[int]:
    gaps = [gap for gap in [1, 4, 10, 23, 57, 132, 301, 701] if gap < n]
    while int(gaps[-1] * 2.25) < n:
        gaps.append(int(gaps[-1] * 2.25))
    return sorted(gaps, reverse=True)


def tokuda_gaps(n: int) -> list[int]:
    gaps, index = [], 1
    while True:
        gap = int(math.ceil((9.0 * (9.0 / 4.0) ** (index - 1) - 4.0) / 5.0))
        if gap >= n:
            break
        gaps.append(gap)
        index += 1
    return sorted(set(gaps) | {1}, reverse=True)


def knuth_gaps(n: int) -> list[int]:
    """Knuth's sequence: (3^k - 1) / 2."""
    gaps, index = [], 1
    while True:
        gap = (3**index - 1) // 2
        if gap >= n:
            break
        gaps.append(gap)
        index += 1
    return sorted(set(gaps) | {1}, reverse=True)


def hibbard_gaps(n: int) -> list[int]:
    """Hibbard's sequence: 2^k - 1."""
    gaps, index = [], 1
    while True:
        gap = 2**index - 1
        if gap >= n:
            break
        gaps.append(gap)
        index += 1
    return sorted(set(gaps) | {1}, reverse=True)


def shell_gaps(n: int) -> list[int]:
    """Shell's original halving sequence."""
    gaps, gap = [], n // 2
    while gap >= 1:
        gaps.append(gap)
        gap //= 2
    return sorted(set(gaps) | {1}, reverse=True)


def pratt_gaps(n: int) -> list[int]:
    """Pratt's 3-smooth sequence: every 2^p 3^q below n."""
    gaps, power_two = set(), 1
    while power_two < n:
        power_three = power_two
        while power_three < n:
            gaps.add(power_three)
            power_three *= 3
        power_two *= 2
    return sorted(gaps, reverse=True)


BASELINES = {
    "Sedgewick": sedgewick_gaps,
    "Ciura": ciura_gaps,
    "Tokuda": tokuda_gaps,
    "Knuth": knuth_gaps,
    "Hibbard": hibbard_gaps,
    "Shell": shell_gaps,
    "Pratt": pratt_gaps,
}


class ShellsortEnv:
    """Environment whose observation is exactly K inversion rates plus two controls."""

    def __init__(self, n: int, max_passes: int = 12, step_penalty: float = 0.005,
                 seed: int | None = None, family_names: tuple[str, ...] = PAPER_FAMILY_NAMES):
        self.n = n
        self.max_passes = max_passes
        self.step_penalty = step_penalty
        self.strides = log_strides(n)
        self.candidate_gaps = make_candidate_gaps(n)
        self.obs_dim = len(self.strides) + 2
        self.rng = np.random.default_rng(seed)
        if not family_names or any(name not in FAMILIES for name in family_names):
            raise ValueError("family_names must contain known array generators")
        self.family_names = tuple(family_names)
        self.array: np.ndarray | None = None
        self.last_gap = n
        self.pass_count = 0
        self.total_comparisons = 0
        self.gaps: list[int] = []

    def reset(self, array: np.ndarray | None = None, family: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        if array is None:
            family = str(self.rng.choice(self.family_names))
            array = FAMILIES[family](self.n, self.rng)
        self.array = array.astype(np.float64, copy=True)
        self.last_gap, self.pass_count, self.total_comparisons = self.n, 0, 0
        self.gaps = []
        return self._observation(), self._mask()

    def _observation(self) -> np.ndarray:
        assert self.array is not None
        controls = np.asarray(
            [1.0 - self.pass_count / self.max_passes, self.last_gap / self.n], dtype=np.float32
        )
        return np.concatenate([inversion_spectrum(self.array, self.strides), controls])

    def _mask(self) -> np.ndarray:
        return np.asarray([gap < self.last_gap for gap in self.candidate_gaps], dtype=bool)

    def step(self, action: int) -> tuple[np.ndarray, np.ndarray, float, bool, dict]:
        assert self.array is not None
        gap = self.candidate_gaps[action]
        if gap >= self.last_gap:
            raise ValueError("actions must choose a strictly decreasing gap")
        comparisons = shellsort_pass(self.array, gap)
        self.total_comparisons += comparisons
        self.pass_count += 1
        self.last_gap = gap
        self.gaps.append(gap)
        is_sorted = bool(np.all(self.array[:-1] <= self.array[1:]))
        done = gap == 1 or is_sorted or self.pass_count >= self.max_passes
        reward = -comparisons / (self.n * math.log2(self.n)) - self.step_penalty
        if done and not is_sorted:
            reward -= 1.0
        info = {"comparisons": self.total_comparisons, "sorted": is_sorted, "gaps": list(self.gaps)}
        return self._observation(), self._mask(), reward, done, info


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.actor = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, action_dim))
        self.critic = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, observations: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(observations)
        logits = self.actor(hidden).masked_fill(~masks, -1e8)
        return logits, self.critic(hidden).squeeze(-1)

    def act(self, observations: torch.Tensor, masks: torch.Tensor, greedy: bool = False):
        logits, values = self(observations, masks)
        distribution = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if greedy else distribution.sample()
        return actions, distribution.log_prob(actions), values, distribution.entropy()


@dataclass(frozen=True)
class PPOConfig:
    n: int = 256
    max_passes: int = 12
    num_envs: int = 32
    rollout_steps: int = 64
    hidden: int = 128
    learning_rate: float = 3e-4
    entropy_start: float = 0.05
    entropy_end: float = 0.005
    step_penalty: float = 0.005
    update_epochs: int = 4
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    family_names: tuple[str, ...] = PAPER_FAMILY_NAMES


class PPOTrainer:
    def __init__(self, config: PPOConfig, total_updates: int, seed: int, device: str = "cpu"):
        self.config, self.total_updates = config, total_updates
        self.device = torch.device(device)
        self.envs = [
            ShellsortEnv(
                config.n,
                config.max_passes,
                config.step_penalty,
                seed + 1009 * index,
                config.family_names,
            )
            for index in range(config.num_envs)
        ]
        self.network = ActorCritic(self.envs[0].obs_dim, len(self.envs[0].candidate_gaps), config.hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.learning_rate)
        self.observations = np.stack([env.reset()[0] for env in self.envs])
        self.masks = np.stack([env._mask() for env in self.envs])

    def train_step(self, update: int) -> dict[str, float]:
        steps, environments = self.config.rollout_steps, self.config.num_envs
        observation_dim, action_dim = self.observations.shape[1], self.masks.shape[1]
        observations = np.empty((steps, environments, observation_dim), dtype=np.float32)
        masks = np.empty((steps, environments, action_dim), dtype=bool)
        actions = np.empty((steps, environments), dtype=np.int64)
        log_probs = np.empty((steps, environments), dtype=np.float32)
        values = np.empty((steps, environments), dtype=np.float32)
        rewards = np.empty((steps, environments), dtype=np.float32)
        dones = np.empty((steps, environments), dtype=np.float32)

        for step in range(steps):
            observations[step], masks[step] = self.observations, self.masks
            with torch.no_grad():
                chosen, log_probability, value, _ = self.network.act(
                    torch.as_tensor(self.observations, dtype=torch.float32, device=self.device),
                    torch.as_tensor(self.masks, dtype=torch.bool, device=self.device),
                )
            actions[step], log_probs[step], values[step] = chosen.cpu().numpy(), log_probability.cpu().numpy(), value.cpu().numpy()
            for index, env in enumerate(self.envs):
                new_observation, new_mask, reward, done, _ = env.step(int(actions[step, index]))
                rewards[step, index], dones[step, index] = reward, float(done)
                if done:
                    new_observation, new_mask = env.reset()
                self.observations[index], self.masks[index] = new_observation, new_mask

        with torch.no_grad():
            _, bootstrap = self.network(
                torch.as_tensor(self.observations, dtype=torch.float32, device=self.device),
                torch.as_tensor(self.masks, dtype=torch.bool, device=self.device),
            )
        advantages = np.zeros_like(rewards)
        carry = np.zeros(environments, dtype=np.float32)
        for step in reversed(range(steps)):
            next_value = bootstrap.cpu().numpy() if step == steps - 1 else values[step + 1]
            active = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * next_value * active - values[step]
            carry = delta + self.config.gamma * self.config.gae_lambda * active * carry
            advantages[step] = carry

        count = steps * environments
        batch = {
            "observations": torch.as_tensor(observations.reshape(count, -1), dtype=torch.float32, device=self.device),
            "masks": torch.as_tensor(masks.reshape(count, -1), dtype=torch.bool, device=self.device),
            "actions": torch.as_tensor(actions.reshape(count), dtype=torch.long, device=self.device),
            "old_log_probs": torch.as_tensor(log_probs.reshape(count), dtype=torch.float32, device=self.device),
            "advantages": torch.as_tensor(advantages.reshape(count), dtype=torch.float32, device=self.device),
            "returns": torch.as_tensor((advantages + values).reshape(count), dtype=torch.float32, device=self.device),
        }
        batch["advantages"] = (batch["advantages"] - batch["advantages"].mean()) / (batch["advantages"].std() + 1e-8)
        fraction = update / self.total_updates
        entropy = self.config.entropy_start + fraction * (self.config.entropy_end - self.config.entropy_start)
        learning_rate = self.config.learning_rate * (1.0 - 0.7 * fraction)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        policy_losses, value_losses = [], []
        for _ in range(self.config.update_epochs):
            indices = torch.randperm(count, device=self.device)
            for start in range(0, count, self.config.minibatch_size):
                subset = indices[start:start + self.config.minibatch_size]
                logits, predicted_values = self.network(batch["observations"][subset], batch["masks"][subset])
                distribution = Categorical(logits=logits)
                ratio = (distribution.log_prob(batch["actions"][subset]) - batch["old_log_probs"][subset]).exp()
                clipped = ratio.clamp(1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon)
                policy_loss = -torch.minimum(ratio * batch["advantages"][subset], clipped * batch["advantages"][subset]).mean()
                value_loss = 0.5 * (predicted_values - batch["returns"][subset]).pow(2).mean()
                loss = policy_loss + self.config.value_coefficient * value_loss - entropy * distribution.entropy().mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                policy_losses.append(float(policy_loss.detach()))
                value_losses.append(float(value_loss.detach()))
        return {"policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(value_losses))}


def run_policy(network: ActorCritic, n: int, max_passes: int, array: np.ndarray, device: str = "cpu") -> dict:
    environment = ShellsortEnv(n=n, max_passes=max_passes, step_penalty=0.0)
    observation, mask = environment.reset(array=array)
    done = False
    while not done:
        with torch.no_grad():
            action, _, _, _ = network.act(
                torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0),
                greedy=True,
            )
        observation, mask, _, done, info = environment.step(int(action.item()))
    return info
