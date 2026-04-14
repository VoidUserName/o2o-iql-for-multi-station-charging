"""Replay helpers for mixed offline-online IQL updates."""
from __future__ import annotations

import numpy as np

from train.iql.data import TransitionDataset


class ReplayBuffer:
    """Simple ring buffer for online transitions."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.action_masks = np.zeros((self.capacity, self.act_dim), dtype=np.uint8)
        self._size = 0
        self._pos = 0

    def __len__(self) -> int:
        return int(self._size)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
    ) -> None:
        self.observations[self._pos] = np.asarray(observation, dtype=np.float32)
        self.actions[self._pos] = int(action)
        self.rewards[self._pos] = float(reward)
        self.next_observations[self._pos] = np.asarray(next_observation, dtype=np.float32)
        self.dones[self._pos] = float(done)
        self.action_masks[self._pos] = np.asarray(action_mask, dtype=np.uint8)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if len(self) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        idx = rng.integers(0, len(self), size=int(batch_size))
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones": self.dones[idx],
            "action_masks": self.action_masks[idx],
        }


def sample_mixed_batch(
    offline_dataset: TransitionDataset,
    online_buffer: ReplayBuffer,
    batch_size: int,
    offline_ratio: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Sample a batch that keeps offline support while incorporating online data."""
    if len(offline_dataset) == 0 and len(online_buffer) == 0:
        raise ValueError("At least one replay source must be non-empty.")

    if len(online_buffer) == 0:
        return offline_dataset.sample(batch_size=batch_size, rng=rng)
    if len(offline_dataset) == 0:
        return online_buffer.sample(batch_size=batch_size, rng=rng)

    offline_count = int(round(float(offline_ratio) * int(batch_size)))
    offline_count = max(0, min(int(batch_size), offline_count))
    online_count = int(batch_size) - offline_count

    if offline_count == 0:
        return online_buffer.sample(batch_size=batch_size, rng=rng)
    if online_count == 0:
        return offline_dataset.sample(batch_size=batch_size, rng=rng)

    offline_batch = offline_dataset.sample(batch_size=offline_count, rng=rng)
    online_batch = online_buffer.sample(batch_size=online_count, rng=rng)
    return {
        key: np.concatenate([offline_batch[key], online_batch[key]], axis=0)
        for key in offline_batch
    }
