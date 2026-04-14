"""Adaptive Behavior Cloning callback for MaskablePPO online fine-tuning.

Implements the PD-controller-based adaptive weighting of the BC loss from:

    "Adaptive Behavior Cloning Regularization for Stable
     Offline-to-Online Reinforcement Learning"
    Zhao et al., NeurIPS Offline RL Workshop 2021.

The paper's Equation 2:
    Δ(α_online) = K_P · (R_avg − R_target) + K_D · max(0, R_avg − R_current)

where α_online ∈ [0, α_offline].

Adaptation for discrete-action MaskablePPO:
  - BC loss = − α · E[log π(a_expert | s)]   (cross-entropy with expert demos)
  - Applied as auxiliary gradient steps between PPO rollouts.
  - The policy's PPO loss is unchanged; BC is a parallel update with its own
    Adam optimiser (separate learning rate).

Usage (standalone fine-tuning script):
    python -m train.imitation.adaptive_bc \\
        --pretrained_model train/imitation/checkpoints/bc_pretrained \\
        --train_data_dir   data/train_dataset/normal

Usage (as a callback inside ppo_trainer):
    from train.imitation.adaptive_bc import AdaptiveBCCallback, load_demo_buffer
    obs, acts, masks = load_demo_buffer(paired_data, n_bins, max_queue_len)
    callback = AdaptiveBCCallback(obs, acts, masks, alpha_offline=0.4, r_target=0.0)
    model.learn(total_timesteps=..., callback=[..., callback])
"""
from __future__ import annotations

import argparse
import io
import os
import zipfile
from collections import deque
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from stable_baselines3.common import base_class as sb3_base_class
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.save_util import get_device, json_to_data, open_path
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from data.env_data import CAPACITY, MIN_SEG, TRAVEL_MATRIX
from envs.maskable_actions import no_split_action_int
from simulator.orchestrator import load_demand_vehicles_from_csv
from train.finetune.ppo_trainer import FlatObsWrapper, make_charging_env
from train.imitation.bc_trainer import (
    collect_demonstrations,
    load_paired_dataset,
)


# ---------------------------------------------------------------------------
# Demo-buffer loader (thin wrapper for external callers)
# ---------------------------------------------------------------------------

def load_demo_buffer(
    paired_data: list[tuple[list, dict]],
    n_bins: int = 21,
    max_queue_len: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (obs, actions, masks) from expert solution pairs.

    Wraps ``collect_demonstrations`` from bc_trainer so callers don't need to
    import that module directly.
    """
    return collect_demonstrations(
        paired_data=paired_data,
        n_bins=n_bins,
        max_queue_len=max_queue_len,
        seed=seed,
    )


def _load_from_zip_file_via_bytes(
    load_path: str | Path,
    load_data: bool = True,
    custom_objects: dict[str, Any] | None = None,
    device: torch.device | str = "auto",
    verbose: int = 0,
    print_system_info: bool = False,
):
    """SB3-compatible loader that reads nested .pth files via BytesIO.

    On this Windows/PyTorch setup, ``torch.load()`` can fail when handed the
    ``ZipExtFile`` stream returned by ``archive.open()``. Reading the payload
    into memory first avoids that issue while keeping the checkpoint format
    unchanged.
    """
    del print_system_info  # Unused in this compatibility path.
    file = open_path(load_path, "r", verbose=verbose, suffix="zip")
    device = get_device(device=device)

    try:
        with zipfile.ZipFile(file) as archive:
            namelist = archive.namelist()
            data = None
            pytorch_variables = None
            params = {}

            if "data" in namelist and load_data:
                json_data = archive.read("data").decode()
                data = json_to_data(json_data, custom_objects=custom_objects)

            pth_files = [
                file_name
                for file_name in namelist
                if os.path.splitext(file_name)[1] == ".pth"
            ]
            for file_path in pth_files:
                payload = io.BytesIO(archive.read(file_path))
                th_object = torch.load(payload, map_location=device, weights_only=True)
                if file_path in {"pytorch_variables.pth", "tensors.pth"}:
                    pytorch_variables = th_object
                else:
                    params[os.path.splitext(file_path)[0]] = th_object
    finally:
        if isinstance(load_path, (str, Path)):
            file.close()

    return data, params, pytorch_variables


def _load_pretrained_model_compat(
    model_path: str,
    env,
    tensorboard_log: str,
    verbose: int,
) -> MaskablePPO:
    """Load a MaskablePPO checkpoint, falling back for zip-stream issues."""
    try:
        return MaskablePPO.load(
            model_path,
            env=env,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
        )
    except RuntimeError as exc:
        if ".data/serialization_id" not in str(exc):
            raise

        print("  Falling back to compatibility checkpoint loader")
        original_loader = sb3_base_class.load_from_zip_file
        sb3_base_class.load_from_zip_file = _load_from_zip_file_via_bytes
        try:
            return MaskablePPO.load(
                model_path,
                env=env,
                tensorboard_log=tensorboard_log,
                verbose=verbose,
            )
        finally:
            sb3_base_class.load_from_zip_file = original_loader


def _resolve_tensorboard_log_dir(log_dir: str) -> str | None:
    """Return a TensorBoard log dir only when tensorboard is installed."""
    if find_spec("tensorboard") is None:
        print("  TensorBoard not installed; continuing without tensorboard logging")
        return None
    return log_dir


def _resolve_progress_bar_flag() -> bool:
    """Enable SB3 progress bars only when their optional deps are installed."""
    has_progress_bar_deps = find_spec("tqdm") is not None and find_spec("rich") is not None
    if not has_progress_bar_deps:
        print("  tqdm/rich not installed; continuing without progress bar")
    return has_progress_bar_deps


# ---------------------------------------------------------------------------
# Adaptive BC callback
# ---------------------------------------------------------------------------

class AdaptiveBCCallback(BaseCallback):
    """Adaptive BC regularization callback (Zhao et al., 2021) for MaskablePPO.

    After each PPO rollout collection, performs ``bc_gradient_steps`` auxiliary
    gradient steps using the BC loss weighted by the current adaptive α.
    α is updated after every completed episode via the PD controller (Eq. 2).

    Args:
        demo_observations:   (N, obs_dim) float32 expert observations.
        demo_actions:        (N,) int64 expert actions.
        demo_masks:          (N, act_dim) float32 action masks at demo steps.
        alpha_offline:       BC weight during offline pre-training — serves as
                             the upper bound for α_online.  Default: 0.4.
        r_target:            Target episodic return (higher = better).
                             Set to expected episode reward of the expert.
        kp:                  Proportional gain: controls how fast α decreases
                             when performance is below target.  Default: 3e-5.
        kd:                  Derivative gain: controls how fast α increases
                             when performance drops (stabilisation).  Default: 1e-4.
        avg_window:          Number of recent episodes used to compute R_avg.
        bc_batch_size:       Mini-batch size for each BC gradient step.
        bc_gradient_steps:   Number of BC gradient steps per PPO rollout.
        bc_lr:               Learning rate for the BC auxiliary Adam optimiser.
        verbose:             0 = silent, 1 = per-episode α log, 2 = per-rollout.
    """

    def __init__(
        self,
        demo_observations: np.ndarray,
        demo_actions: np.ndarray,
        demo_masks: np.ndarray,
        alpha_offline: float = 0.4,
        r_target: float = 0.0,
        kp: float = 3e-5,
        kd: float = 1e-4,
        avg_window: int = 10,
        bc_batch_size: int = 256,
        bc_gradient_steps: int = 10,
        bc_lr: float = 3e-4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.demo_obs     = demo_observations
        self.demo_actions = demo_actions
        self.demo_masks   = demo_masks
        self._n_demos     = len(demo_observations)

        self.alpha_offline     = float(alpha_offline)
        self.alpha_online      = float(alpha_offline)   # starts at α_offline
        self.r_target          = float(r_target)
        self.kp                = float(kp)
        self.kd                = float(kd)
        self.avg_window        = int(avg_window)
        self.bc_batch_size     = int(bc_batch_size)
        self.bc_gradient_steps = int(bc_gradient_steps)
        self.bc_lr             = float(bc_lr)

        self._ep_returns: deque[float] = deque(maxlen=avg_window)
        self._r_avg = 0.0
        self._bc_optimizer: Optional[torch.optim.Optimizer] = None

    # ------------------------------------------------------------------
    # SB3 callback hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        self._bc_optimizer = torch.optim.Adam(
            self.model.policy.parameters(), lr=self.bc_lr
        )
        if self.verbose >= 1:
            print(
                f"[AdaptiveBC] init: alpha={self.alpha_online:.4f}  "
                f"r_target={self.r_target:.2f}  Kp={self.kp}  Kd={self.kd}"
            )

    def _on_step(self) -> bool:
        """Detect episode ends and update α via the PD controller."""
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                r_current = float(ep_info["r"])
                self._ep_returns.append(r_current)
                self._update_alpha(r_current)
                if self.verbose >= 1:
                    print(
                        f"[AdaptiveBC] ep_return={r_current:.3f}  "
                        f"r_avg={self._r_avg:.3f}  alpha={self.alpha_online:.5f}"
                    )
        return True

    def _on_rollout_end(self) -> None:
        """Perform BC auxiliary gradient steps after every PPO rollout."""
        self._do_bc_steps()

    # ------------------------------------------------------------------
    # Adaptive α — Eq. 2 from Zhao et al. 2021
    # ------------------------------------------------------------------

    def _update_alpha(self, r_current: float) -> None:
        """Update α_online with the PD controller.

        Δα = K_P · (R_avg − R_target) + K_D · max(0, R_avg − R_current)

        When performance is below target (R_avg < R_target):
          → proportional term is negative → α decreases → more RL exploration.
        When current return drops below average (R_current < R_avg):
          → derivative term is positive → α increases → more BC → stabilise.
        """
        if not self._ep_returns:
            self._r_avg = r_current
            return

        r_avg = float(np.mean(self._ep_returns))
        delta = (
            self.kp * (r_avg - self.r_target)
            + self.kd * max(0.0, r_avg - r_current)
        )
        self.alpha_online = float(
            np.clip(self.alpha_online + delta, 0.0, self.alpha_offline)
        )
        self._r_avg = r_avg

    # ------------------------------------------------------------------
    # BC gradient steps
    # ------------------------------------------------------------------

    def _do_bc_steps(self) -> None:
        """Sample demo mini-batches; apply weighted BC loss to the policy."""
        if self.alpha_online < 1e-8:
            if self.verbose >= 2:
                print("[AdaptiveBC] alpha≈0 — skipping BC steps")
            return

        policy = self.model.policy
        policy.set_training_mode(True)

        total_bc_loss = 0.0
        for _ in range(self.bc_gradient_steps):
            idx = np.random.randint(0, self._n_demos, size=self.bc_batch_size)
            obs_b  = torch.FloatTensor(self.demo_obs[idx]).to(self.model.device)
            act_b  = torch.LongTensor(self.demo_actions[idx]).to(self.model.device)
            mask_b = self.demo_masks[idx]   # (B, act_dim) np.ndarray

            _, log_prob, _ = policy.evaluate_actions(obs_b, act_b, action_masks=mask_b)
            bc_loss = -self.alpha_online * log_prob.mean()

            self._bc_optimizer.zero_grad()
            bc_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            self._bc_optimizer.step()
            total_bc_loss += bc_loss.item()

        policy.set_training_mode(False)

        if self.verbose >= 2:
            avg = total_bc_loss / self.bc_gradient_steps
            print(
                f"[AdaptiveBC] bc_steps={self.bc_gradient_steps}  "
                f"avg_bc_loss={avg:.4f}  alpha={self.alpha_online:.5f}"
            )


# ---------------------------------------------------------------------------
# Standalone fine-tuning entry-point
# ---------------------------------------------------------------------------

def run_adaptive_bc_finetuning(args: argparse.Namespace) -> None:
    """Load a BC pre-trained model and fine-tune with Adaptive BC + MaskablePPO."""
    print("=== Adaptive BC Online Fine-tuning ===")

    # -- Demo buffer from bc_dataset -----------------------------------------
    paired_data = load_paired_dataset(
        demand_dir=args.demand_dir,
        solution_dir=args.solution_dir,
    )
    demo_obs, demo_actions, demo_masks = load_demo_buffer(
        paired_data=paired_data,
        n_bins=args.n_bins,
        max_queue_len=args.max_queue_len,
        seed=args.seed,
    )

    # -- Training environments ------------------------------------------------
    train_episode_bank = [
        load_demand_vehicles_from_csv(str(f))
        for f in sorted(Path(args.train_data_dir).glob("*.csv"))
    ]
    if not train_episode_bank:
        raise FileNotFoundError(f"No CSV episodes in '{args.train_data_dir}'")
    print(f"  Train episodes: {len(train_episode_bank)}")

    env_fns = [
        make_charging_env(
            episode_bank=train_episode_bank,
            n_bins=args.n_bins,
            max_queue_len=args.max_queue_len,
            invalid_action_penalty=args.invalid_action_penalty,
            seed=args.seed + i,
        )
        for i in range(args.n_envs)
    ]
    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_env = VecCls(env_fns)

    eval_env_fns = [
        make_charging_env(
            episode_bank=train_episode_bank,
            n_bins=args.n_bins,
            max_queue_len=args.max_queue_len,
            seed=args.seed + 9999,
        )
    ]
    eval_env = VecCls(eval_env_fns)

    # -- Load pre-trained model -----------------------------------------------
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)
    tensorboard_log = _resolve_tensorboard_log_dir(args.log_dir)

    if args.pretrained_model:
        print(f"  Loading pre-trained model from '{args.pretrained_model}'")
        model = _load_pretrained_model_compat(
            args.pretrained_model,
            env=vec_env,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
    else:
        print("  No pre-trained model supplied — training from scratch")
        model = MaskablePPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            tensorboard_log=tensorboard_log,
            verbose=1,
            seed=args.seed,
        )

    # -- Adaptive BC callback -------------------------------------------------
    adaptive_bc = AdaptiveBCCallback(
        demo_observations=demo_obs,
        demo_actions=demo_actions,
        demo_masks=demo_masks,
        alpha_offline=args.alpha_offline,
        r_target=args.r_target,
        kp=args.kp,
        kd=args.kd,
        avg_window=args.avg_window,
        bc_batch_size=args.bc_batch_size,
        bc_gradient_steps=args.bc_gradient_steps,
        bc_lr=args.bc_lr,
        verbose=1,
    )

    eval_freq = max(args.eval_freq // args.n_envs, 1)
    ckpt_freq = max(args.checkpoint_freq // args.n_envs, 1)

    callbacks = [
        adaptive_bc,
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=args.save_path,
            log_path=args.log_dir,
            eval_freq=eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=args.save_path,
            name_prefix="adaptive_bc_ppo",
            verbose=1,
        ),
    ]

    progress_bar = _resolve_progress_bar_flag()
    print(f"  Fine-tuning for {args.total_timesteps:,} timesteps")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=progress_bar,
        reset_num_timesteps=False,  # continue from pre-trained step count
    )

    final_path = os.path.join(args.save_path, "adaptive_bc_final")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")
    vec_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Adaptive BC fine-tuning (Zhao et al. 2021) for EV Charging"
    )

    # Demo data (bc_dataset)
    p.add_argument("--demand_dir",   type=str,   default="data/bc_dataset/demand")
    p.add_argument("--solution_dir", type=str,   default="data/bc_dataset/solutions")

    # Online training data
    p.add_argument("--train_data_dir", type=str, default="data/train_dataset/normal")

    # Env
    p.add_argument("--n_bins",                  type=int,   default=21)
    p.add_argument("--max_queue_len",           type=int,   default=10)
    p.add_argument("--invalid_action_penalty",  type=float, default=0.0)

    # Pre-trained model (optional)
    p.add_argument("--pretrained_model", type=str, default="",
                   help="Path to BC pre-trained .zip (without extension)")

    # PPO hyperparameters (only used when pretrained_model is not given)
    p.add_argument("--learning_rate",   type=float, default=3e-4)
    p.add_argument("--n_steps",         type=int,   default=2048)
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--n_epochs",        type=int,   default=10)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae_lambda",      type=float, default=0.95)
    p.add_argument("--clip_range",      type=float, default=0.2)
    p.add_argument("--ent_coef",        type=float, default=0.01)
    p.add_argument("--vf_coef",         type=float, default=0.5)
    p.add_argument("--max_grad_norm",   type=float, default=0.5)

    # Adaptive BC hyperparameters
    p.add_argument("--alpha_offline",       type=float, default=0.4,
                   help="BC weight during offline pre-training (upper bound for α_online)")
    p.add_argument("--r_target",            type=float, default=0.0,
                   help="Target episodic return; expert-level performance")
    p.add_argument("--kp",                  type=float, default=3e-5,
                   help="PD proportional gain (controls α decrease rate)")
    p.add_argument("--kd",                  type=float, default=1e-4,
                   help="PD derivative gain (controls α increase on performance drop)")
    p.add_argument("--avg_window",          type=int,   default=10,
                   help="Episode window for computing R_avg")
    p.add_argument("--bc_batch_size",       type=int,   default=256)
    p.add_argument("--bc_gradient_steps",   type=int,   default=10,
                   help="BC gradient steps per PPO rollout")
    p.add_argument("--bc_lr",               type=float, default=3e-4,
                   help="Learning rate for the BC auxiliary optimiser")

    # Training
    p.add_argument("--total_timesteps",     type=int,   default=500_000)
    p.add_argument("--n_envs",              type=int,   default=4)
    p.add_argument("--seed",                type=int,   default=42)

    # Eval / saving
    p.add_argument("--eval_freq",           type=int,   default=25_000)
    p.add_argument("--n_eval_episodes",     type=int,   default=10)
    p.add_argument("--checkpoint_freq",     type=int,   default=50_000)
    p.add_argument("--save_path",           type=str,   default="train/imitation/checkpoints")
    p.add_argument("--log_dir",             type=str,   default="train/imitation/logs")

    return p.parse_args()


if __name__ == "__main__":
    run_adaptive_bc_finetuning(parse_args())
