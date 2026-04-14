"""Balanced dual-buffer replay for offline-to-online IQL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train.iql.data import TransitionDataset
from train.iql.replay import ReplayBuffer


def _build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    dims = [int(input_dim), *[int(dim) for dim in hidden_dims]]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], int(output_dim)))
    return nn.Sequential(*layers)


@dataclass(slots=True)
class PriorityRefreshStats:
    step: int
    classifier_loss: float
    entropy: float
    effective_sample_size: float
    top_priority: float
    online_buffer_size: int


class DensityRatioEstimator:
    """Binary classifier used to score how close offline samples are to online data."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        learning_rate: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.device = torch.device(device)
        self.network = _build_mlp(obs_dim + act_dim, hidden_dims, 1).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=float(learning_rate))

    def _encode(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
    ) -> torch.Tensor:
        obs_t = torch.as_tensor(observations, device=self.device, dtype=torch.float32)
        act_t = torch.as_tensor(actions, device=self.device, dtype=torch.long)
        act_one_hot = F.one_hot(act_t, num_classes=self.act_dim).to(torch.float32)
        return torch.cat([obs_t, act_one_hot], dim=1)

    def fit(
        self,
        offline_dataset: TransitionDataset,
        online_buffer: ReplayBuffer,
        rng: np.random.Generator,
        n_steps: int,
        batch_size: int,
    ) -> float:
        """Train the classifier to distinguish online vs. offline samples."""
        if len(online_buffer) == 0:
            return 0.0

        losses: list[float] = []
        online_batch_size = max(1, int(batch_size) // 2)
        offline_batch_size = max(1, int(batch_size) - online_batch_size)
        criterion = nn.BCEWithLogitsLoss()

        for _ in range(int(n_steps)):
            offline_batch = offline_dataset.sample(batch_size=offline_batch_size, rng=rng)
            online_batch = online_buffer.sample(batch_size=online_batch_size, rng=rng)

            offline_inputs = self._encode(
                observations=offline_batch["observations"],
                actions=offline_batch["actions"],
            )
            online_inputs = self._encode(
                observations=online_batch["observations"],
                actions=online_batch["actions"],
            )
            inputs = torch.cat([offline_inputs, online_inputs], dim=0)
            labels = torch.cat(
                [
                    torch.zeros(offline_inputs.shape[0], device=self.device, dtype=torch.float32),
                    torch.ones(online_inputs.shape[0], device=self.device, dtype=torch.float32),
                ],
                dim=0,
            )

            logits = self.network(inputs).squeeze(1)
            loss = criterion(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))

        return float(np.mean(losses)) if losses else 0.0

    def score_offline_dataset(
        self,
        offline_dataset: TransitionDataset,
        chunk_size: int = 2048,
        max_ratio: float = 50.0,
    ) -> np.ndarray:
        """Return density-ratio-like scores for every offline sample."""
        self.network.eval()
        scores: list[np.ndarray] = []
        max_ratio = float(max_ratio)
        eps = 1e-6

        with torch.no_grad():
            for start in range(0, len(offline_dataset), int(chunk_size)):
                stop = min(start + int(chunk_size), len(offline_dataset))
                features = self._encode(
                    observations=offline_dataset.observations[start:stop],
                    actions=offline_dataset.actions[start:stop],
                )
                probs = torch.sigmoid(self.network(features).squeeze(1))
                odds = probs / (1.0 - probs + eps)
                scores.append(odds.clamp(min=eps, max=max_ratio).cpu().numpy())

        self.network.train()
        return np.concatenate(scores, axis=0)


class PrioritizedOfflineBuffer:
    """Fixed offline dataset with refreshable non-uniform sampling priorities."""

    def __init__(
        self,
        dataset: TransitionDataset,
        uniform_floor: float = 0.05,
        priority_temperature: float = 1.0,
    ) -> None:
        self.dataset = dataset
        self.uniform_floor = float(uniform_floor)
        self.priority_temperature = float(priority_temperature)
        self.priorities = np.full(len(dataset), 1.0 / max(len(dataset), 1), dtype=np.float64)

    def __len__(self) -> int:
        return int(len(self.dataset))

    def update_priorities(self, raw_scores: np.ndarray) -> tuple[float, float, float]:
        if len(self.dataset) == 0:
            raise ValueError("Cannot update priorities for an empty offline dataset.")

        scores = np.asarray(raw_scores, dtype=np.float64)
        if scores.shape != (len(self.dataset),):
            raise ValueError("Priority score shape does not match offline dataset length.")
        scores = np.maximum(scores, 1e-8)
        if self.priority_temperature != 1.0:
            scores = np.power(scores, self.priority_temperature)
        probs = scores / scores.sum()
        floor = self.uniform_floor / len(probs)
        probs = (1.0 - self.uniform_floor) * probs + floor
        probs = probs / probs.sum()
        self.priorities = probs

        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        ess = float(1.0 / np.square(probs).sum())
        top_priority = float(probs.max())
        return entropy, ess, top_priority

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        if len(self.dataset) == 0:
            raise ValueError("Cannot sample from an empty offline dataset.")
        idx = rng.choice(len(self.dataset), size=int(batch_size), replace=True, p=self.priorities)
        return {
            "observations": self.dataset.observations[idx],
            "actions": self.dataset.actions[idx],
            "rewards": self.dataset.rewards[idx],
            "next_observations": self.dataset.next_observations[idx],
            "dones": self.dataset.dones[idx],
            "action_masks": self.dataset.action_masks[idx],
        }


class BalancedReplayManager:
    """Dual-buffer sampler with prioritized offline replay and near-on-policy online replay."""

    def __init__(
        self,
        offline_dataset: TransitionDataset,
        obs_dim: int,
        act_dim: int,
        online_buffer_size: int = 20_000,
        online_sample_prob: float = 0.5,
        min_online_samples: int = 2_000,
        priority_refresh_freq: int = 5_000,
        priority_model_steps: int = 100,
        priority_batch_size: int = 512,
        priority_model_lr: float = 1e-3,
        priority_uniform_floor: float = 0.05,
        priority_temperature: float = 1.0,
        priority_max_ratio: float = 50.0,
        device: str = "cpu",
    ) -> None:
        self.offline_buffer = PrioritizedOfflineBuffer(
            dataset=offline_dataset,
            uniform_floor=priority_uniform_floor,
            priority_temperature=priority_temperature,
        )
        self.online_buffer = ReplayBuffer(
            capacity=online_buffer_size,
            obs_dim=obs_dim,
            act_dim=act_dim,
        )
        self.online_sample_prob = float(online_sample_prob)
        self.min_online_samples = int(min_online_samples)
        self.priority_refresh_freq = int(priority_refresh_freq)
        self.priority_model_steps = int(priority_model_steps)
        self.priority_batch_size = int(priority_batch_size)
        self.priority_max_ratio = float(priority_max_ratio)
        self.density_estimator = DensityRatioEstimator(
            obs_dim=obs_dim,
            act_dim=act_dim,
            learning_rate=priority_model_lr,
            device=device,
        )

    def add_online_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
        action_mask: np.ndarray,
    ) -> None:
        self.online_buffer.add(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            action_mask=action_mask,
        )

    def can_sample_online(self) -> bool:
        return len(self.online_buffer) >= self.min_online_samples

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[dict[str, np.ndarray], str]:
        use_online = self.can_sample_online() and rng.random() < self.online_sample_prob
        if use_online:
            return self.online_buffer.sample(batch_size=batch_size, rng=rng), "online"
        return self.offline_buffer.sample(batch_size=batch_size, rng=rng), "offline"

    def maybe_refresh_priorities(
        self,
        step: int,
        rng: np.random.Generator,
    ) -> PriorityRefreshStats | None:
        if not self.can_sample_online():
            return None
        if step <= 0 or step % self.priority_refresh_freq != 0:
            return None

        classifier_loss = self.density_estimator.fit(
            offline_dataset=self.offline_buffer.dataset,
            online_buffer=self.online_buffer,
            rng=rng,
            n_steps=self.priority_model_steps,
            batch_size=self.priority_batch_size,
        )
        scores = self.density_estimator.score_offline_dataset(
            offline_dataset=self.offline_buffer.dataset,
            max_ratio=self.priority_max_ratio,
        )
        entropy, ess, top_priority = self.offline_buffer.update_priorities(scores)
        return PriorityRefreshStats(
            step=int(step),
            classifier_loss=float(classifier_loss),
            entropy=entropy,
            effective_sample_size=ess,
            top_priority=top_priority,
            online_buffer_size=int(len(self.online_buffer)),
        )
