"""PopArt variant of MaskablePPO.

Replaces the scalar value head with a PopArt-normalized linear head:

    ỹ = W·h + b                    (raw, normalized output)
    y = σ·ỹ + μ                    (real-scale value)

At the start of each `train()` call, running μ, σ are updated from the
rollout's returns and the head's (W, b) are rescaled so that real-space
outputs are preserved. The value loss is then computed in normalized
space, giving bounded gradients regardless of reward scale.

Rollout-time GAE uses real-scale values (via `predict_values`), so nothing
downstream of the buffer needs to change.
"""
from __future__ import annotations

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.utils import explained_variance


# ---------------------------------------------------------------------------
# PopArt layer
# ---------------------------------------------------------------------------

class PopArtHead(nn.Module):
    """PopArt-normalized linear value head.

    * ``forward(h)`` returns the raw (normalized) scalar ỹ.
    * ``unnormalize(ỹ)`` returns σ·ỹ + μ.
    * ``normalize(y)`` returns (y − μ) / σ.
    * ``update(targets)`` updates running (μ, σ) from a batch of real-scale
      targets and rescales (W, b) so that unnormalize(forward(h)) is
      preserved across the update.
    """

    def __init__(self, in_features: int, beta: float = 3e-4, eps: float = 1e-5):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.beta = float(beta)
        self.eps = float(eps)
        self.register_buffer("mu",    th.zeros(1))
        self.register_buffer("sigma", th.ones(1))
        self.register_buffer("nu",    th.ones(1))  # running E[y²]

    @property
    def in_features(self) -> int:
        return self.linear.in_features

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)

    def unnormalize(self, normalized: th.Tensor) -> th.Tensor:
        return normalized * self.sigma + self.mu

    def normalize(self, real: th.Tensor) -> th.Tensor:
        return (real - self.mu) / self.sigma

    @th.no_grad()
    def update(self, targets: th.Tensor) -> None:
        targets = targets.detach().to(self.mu.device, dtype=th.float32).flatten()
        if targets.numel() == 0:
            return

        old_mu    = self.mu.clone()
        old_sigma = self.sigma.clone()

        batch_mu = targets.mean().view(1)
        batch_nu = (targets * targets).mean().view(1)

        self.mu.mul_(1.0 - self.beta).add_(self.beta * batch_mu)
        self.nu.mul_(1.0 - self.beta).add_(self.beta * batch_nu)
        new_sigma = (self.nu - self.mu * self.mu).clamp_min(self.eps).sqrt()
        self.sigma.copy_(new_sigma)

        # Preserve real-space outputs: σ_new·(W'h + b') + μ_new == σ_old·(Wh + b) + μ_old
        scale = (old_sigma / self.sigma)  # shape (1,)
        self.linear.weight.data.mul_(scale.view(1, 1))
        self.linear.bias.data.copy_(
            (old_sigma * self.linear.bias.data + old_mu - self.mu) / self.sigma
        )


# ---------------------------------------------------------------------------
# Policy with a PopArt value head
# ---------------------------------------------------------------------------

class PopArtMaskablePolicy(MaskableActorCriticPolicy):
    """MaskableActorCriticPolicy whose ``value_net`` is a :class:`PopArtHead`.

    * ``predict_values`` returns **real-scale** values (for GAE/rollout buffer).
    * The inherited ``evaluate_actions`` calls ``self.value_net(latent_vf)``
      and thus returns **normalized** values — consumed by the trainer to
      compute a normalized MSE value loss.
    """

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)

        in_features = self.value_net.in_features
        self.value_net = PopArtHead(in_features).to(self.device)

        # Re-create the optimizer so it tracks the new value_net parameters.
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def predict_values(self, obs) -> th.Tensor:
        features = self.extract_features(obs, self.vf_features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net.unnormalize(self.value_net(latent_vf))


# ---------------------------------------------------------------------------
# Trainer — MaskablePPO with PopArt-normalized value loss
# ---------------------------------------------------------------------------

class PopArtMaskablePPO(MaskablePPO):
    """MaskablePPO whose value loss is computed in PopArt-normalized space."""

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        # ---- PopArt: update running stats once per train() call, using the
        #      real-scale returns from the rollout buffer. After this call
        #      the value head is rescaled so its real-space outputs are
        #      preserved across the μ,σ update.
        pop_head: PopArtHead = self.policy.value_net  # type: ignore[assignment]
        returns_tensor = th.as_tensor(
            self.rollout_buffer.returns, device=self.device, dtype=th.float32
        )
        pop_head.update(returns_tensor)
        pop_mu = pop_head.mu
        pop_sigma = pop_head.sigma

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs: list[float] = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                # `values` here are NORMALIZED because self.value_net is PopArtHead.
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )
                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # Normalize targets into PopArt space for the value loss.
                normalized_returns = ((rollout_data.returns - pop_mu) / pop_sigma).flatten()

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    # old_values were stored in real space; normalize them so
                    # the clip happens in the same space as `values`.
                    old_values_norm = ((rollout_data.old_values - pop_mu) / pop_sigma).flatten()
                    values_pred = old_values_norm + th.clamp(
                        values - old_values_norm, -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(normalized_returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        self.logger.record("train/entropy_loss",         np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss",           np.mean(value_losses))
        self.logger.record("train/approx_kl",            np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction",        np.mean(clip_fractions))
        self.logger.record("train/loss",                 loss.item())
        self.logger.record("train/explained_variance",   explained_var)
        self.logger.record("train/popart_mu",            float(pop_mu.item()))
        self.logger.record("train/popart_sigma",         float(pop_sigma.item()))
        self.logger.record("train/n_updates",            self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range",           clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
