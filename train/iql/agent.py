"""Discrete masked-action IQL agent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def _build_mlp(input_dim: int, hidden_dims: Iterable[int], output_dim: int) -> nn.Sequential:
    dims = [int(input_dim), *[int(dim) for dim in hidden_dims]]
    layers: list[nn.Module] = []
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dims[-1], int(output_dim)))
    return nn.Sequential(*layers)


def sanitize_action_masks(
    action_masks: np.ndarray | torch.Tensor,
    action_dim: int,
) -> torch.Tensor:
    """Ensure categorical masking always has at least one valid action."""
    masks = torch.as_tensor(action_masks, dtype=torch.bool)
    if masks.ndim == 1:
        masks = masks.unsqueeze(0)
    if masks.shape[-1] != action_dim:
        raise ValueError(
            f"Expected action mask width {action_dim}, received {masks.shape[-1]}"
        )
    if masks.numel() == 0:
        raise ValueError("Empty action mask tensor is not supported.")
    no_valid = ~masks.any(dim=1)
    if no_valid.any():
        masks = masks.clone()
        masks[no_valid] = True
    return masks


def _masked_logits(logits: torch.Tensor, action_masks: np.ndarray | torch.Tensor) -> torch.Tensor:
    valid_masks = sanitize_action_masks(action_masks, action_dim=logits.shape[-1]).to(logits.device)
    fill_value = torch.finfo(logits.dtype).min
    return logits.masked_fill(~valid_masks, fill_value)


def _expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return weight * diff.pow(2)


class MaskedPolicyNetwork(nn.Module):
    """Simple MLP policy over a discrete action space with action masks."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.net = _build_mlp(obs_dim, hidden_dims, act_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)

    def distribution(
        self,
        observations: torch.Tensor,
        action_masks: np.ndarray | torch.Tensor,
    ) -> Categorical:
        logits = self.forward(observations)
        return Categorical(logits=_masked_logits(logits, action_masks))

    def log_prob(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        return self.distribution(observations, action_masks).log_prob(actions)

    def act(
        self,
        observation: torch.Tensor,
        action_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> int:
        dist = self.distribution(observation.unsqueeze(0), action_mask.unsqueeze(0))
        if deterministic:
            action = torch.argmax(dist.logits, dim=1)
        else:
            action = dist.sample()
        return int(action.item())


class DiscreteQNetwork(nn.Module):
    """State-action value network that predicts Q(s, a) for all actions."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, act_dim)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class ValueNetwork(nn.Module):
    """Scalar state-value network used by IQL."""

    def __init__(self, obs_dim: int, hidden_dims: Iterable[int]) -> None:
        super().__init__()
        self.net = _build_mlp(obs_dim, hidden_dims, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations).squeeze(-1)


@dataclass(slots=True)
class IQLUpdateMetrics:
    actor_loss: float
    critic_loss: float
    value_loss: float
    mean_advantage: float
    mean_weight: float


class DiscreteIQLAgent:
    """Masked discrete-action IQL with twin critics and a value network."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        discount: float = 0.99,
        expectile: float = 0.7,
        temperature: float = 3.0,
        target_update_rate: float = 5e-3,
        exp_adv_max: float = 100.0,
        device: str = "cpu",
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.learning_rate = float(learning_rate)
        self.discount = float(discount)
        self.expectile = float(expectile)
        self.temperature = float(temperature)
        self.target_update_rate = float(target_update_rate)
        self.exp_adv_max = float(exp_adv_max)
        self.device = torch.device(device)

        self.actor = MaskedPolicyNetwork(obs_dim, act_dim, hidden_dims=self.hidden_dims).to(self.device)
        self.q1 = DiscreteQNetwork(obs_dim, act_dim, hidden_dims=self.hidden_dims).to(self.device)
        self.q2 = DiscreteQNetwork(obs_dim, act_dim, hidden_dims=self.hidden_dims).to(self.device)
        self.target_q1 = DiscreteQNetwork(obs_dim, act_dim, hidden_dims=self.hidden_dims).to(self.device)
        self.target_q2 = DiscreteQNetwork(obs_dim, act_dim, hidden_dims=self.hidden_dims).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden_dims=self.hidden_dims).to(self.device)

        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.learning_rate,
        )
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=self.learning_rate)
        self.update_count = 0

    def _to_tensor(
        self,
        value: np.ndarray | torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.as_tensor(value, device=self.device, dtype=dtype)

    def set_expectile(self, expectile: float) -> None:
        self.expectile = float(expectile)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = float(temperature)

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        with torch.no_grad():
            obs_t = self._to_tensor(observation, dtype=torch.float32)
            mask_t = self._to_tensor(action_mask, dtype=torch.bool)
            return self.actor.act(obs_t, mask_t, deterministic=deterministic)

    def act_ucb(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        ucb_coef: float = 1.0,
    ) -> int:
        """Select action via UCB(Q): argmax_valid [Q_mean + ucb_coef * Q_std]."""
        with torch.no_grad():
            obs_t = self._to_tensor(observation, dtype=torch.float32).unsqueeze(0)
            mask_t = self._to_tensor(action_mask, dtype=torch.bool).unsqueeze(0)
            q1 = self.q1(obs_t)
            q2 = self.q2(obs_t)
            q_mean = (q1 + q2) / 2.0
            q_std = (q1 - q2).abs() / 2.0
            ucb = q_mean + ucb_coef * q_std
            fill_value = torch.finfo(ucb.dtype).min
            ucb = ucb.masked_fill(~mask_t, fill_value)
            return int(ucb.argmax(dim=1).item())

    def update(self, batch: dict[str, np.ndarray | torch.Tensor]) -> IQLUpdateMetrics:
        obs = self._to_tensor(batch["observations"], dtype=torch.float32)
        actions = self._to_tensor(batch["actions"], dtype=torch.long)
        rewards = self._to_tensor(batch["rewards"], dtype=torch.float32)
        next_obs = self._to_tensor(batch["next_observations"], dtype=torch.float32)
        dones = self._to_tensor(batch["dones"], dtype=torch.float32)
        action_masks = self._to_tensor(batch["action_masks"], dtype=torch.bool)

        with torch.no_grad():
            target_q = torch.min(self.target_q1(obs), self.target_q2(obs))
            target_q_a = target_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        value_pred = self.value(obs)
        value_loss = _expectile_loss(target_q_a - value_pred, self.expectile).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        with torch.no_grad():
            next_v = self.value(next_obs)
            q_target = rewards + self.discount * (1.0 - dones) * next_v

        q1_pred = self.q1(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_pred = self.q2(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_loss = ((q1_pred - q_target).pow(2) + (q2_pred - q_target).pow(2)).mean()
        self.q_optimizer.zero_grad()
        critic_loss.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            target_q = torch.min(self.target_q1(obs), self.target_q2(obs))
            target_q_a = target_q.gather(1, actions.unsqueeze(1)).squeeze(1)
            advantages = target_q_a - self.value(obs)
            exp_adv = torch.exp(self.temperature * advantages).clamp(max=self.exp_adv_max)

        log_prob = self.actor.log_prob(obs, actions, action_masks)
        actor_loss = -(exp_adv * log_prob).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self._soft_update_targets()
        self.update_count += 1

        return IQLUpdateMetrics(
            actor_loss=float(actor_loss.item()),
            critic_loss=float(critic_loss.item()),
            value_loss=float(value_loss.item()),
            mean_advantage=float(advantages.mean().item()),
            mean_weight=float(exp_adv.mean().item()),
        )

    def _soft_update_targets(self) -> None:
        tau = self.target_update_rate
        for target_param, param in zip(self.target_q1.parameters(), self.q1.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
        for target_param, param in zip(self.target_q2.parameters(), self.q2.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def save(self, path: str | Path, extra_state: dict | None = None) -> None:
        payload = {
            "config": {
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
                "hidden_dims": list(self.hidden_dims),
                "learning_rate": self.learning_rate,
                "discount": self.discount,
                "expectile": self.expectile,
                "temperature": self.temperature,
                "target_update_rate": self.target_update_rate,
                "exp_adv_max": self.exp_adv_max,
            },
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "value": self.value.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "update_count": self.update_count,
            "extra_state": extra_state or {},
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "DiscreteIQLAgent":
        payload = torch.load(path, map_location=device, weights_only=False)
        config = payload["config"]
        agent = cls(
            obs_dim=int(config["obs_dim"]),
            act_dim=int(config["act_dim"]),
            hidden_dims=tuple(int(dim) for dim in config["hidden_dims"]),
            learning_rate=float(config["learning_rate"]),
            discount=float(config["discount"]),
            expectile=float(config["expectile"]),
            temperature=float(config["temperature"]),
            target_update_rate=float(config["target_update_rate"]),
            exp_adv_max=float(config["exp_adv_max"]),
            device=device,
        )
        agent.actor.load_state_dict(payload["actor"])
        agent.q1.load_state_dict(payload["q1"])
        agent.q2.load_state_dict(payload["q2"])
        agent.target_q1.load_state_dict(payload["target_q1"])
        agent.target_q2.load_state_dict(payload["target_q2"])
        agent.value.load_state_dict(payload["value"])
        agent.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        agent.q_optimizer.load_state_dict(payload["q_optimizer"])
        agent.value_optimizer.load_state_dict(payload["value_optimizer"])
        agent.update_count = int(payload.get("update_count", 0))
        return agent
