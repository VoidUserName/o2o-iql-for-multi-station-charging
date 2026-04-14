"""Maskable-PPO trainer for the MultiStation EV Charging environment.

Usage:
    python -m train.ppo_trainer [--data_dir ...] [--total_timesteps ...]

Dependencies (beyond the project):
    pip install sb3-contrib stable-baselines3 gymnasium
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import is_masking_supported
from sb3_contrib.common.wrappers import ActionMasker

from train.finetune.popart_ppo import PopArtMaskablePPO, PopArtMaskablePolicy

from data.env_data import CAPACITY, MIN_SEG, TRAVEL_MATRIX
from envs.charging_env import (
    EpisodeBankChargingEnv,
    travel_time_fn_from_matrix,
)
from simulator.orchestrator import load_demand_vehicles_from_csv


# ---------------------------------------------------------------------------
# Observation flattening
# ---------------------------------------------------------------------------

class FlatObsWrapper(gym.ObservationWrapper):
    """Flatten the nested Dict/Sequence observation into a 1-D float32 Box.

    The env's observation contains variable-length Sequence spaces
    (queue_waiting_time, queue_demand, downstream_stations) that are
    incompatible with SB3's default MlpPolicy. This wrapper:
      - Pads / truncates queue sequences to ``max_queue_len``.
      - Encodes downstream_stations as a binary mask over all stations.
      - Returns a flat float32 array.
    """

    def __init__(self, env: gym.Env, max_queue_len: int = 10) -> None:
        super().__init__(env)
        self.max_queue_len = int(max_queue_len)
        self.num_stations = int(env.get_wrapper_attr("num_stations"))
        self.station_capacities = list(env.get_wrapper_attr("station_capacities"))
        num_stations = self.num_stations
        capacities = self.station_capacities
        mq = self.max_queue_len

        # Flat dimension breakdown:
        #   sim_state.clock                  : 1
        #   per station (cap + 2*mq)        : varies
        #   metrics (3 * num_stations)       : 21
        #   commitment_features (3*S)        : 21
        #   current_ev (2 + S)               : 9
        #   future_demand (S)                : 7
        sim_dim = (
            1
            + sum(cap + 2 * mq for cap in capacities)
            + 3 * num_stations
        )
        commit_dim = 3 * num_stations
        ev_dim = 2 + num_stations
        future_dim = num_stations

        total_dim = sim_dim + commit_dim + ev_dim + future_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_dim,), dtype=np.float32
        )

    def observation(self, obs: dict) -> np.ndarray:
        num_stations = self.num_stations
        capacities = self.station_capacities
        mq = self.max_queue_len
        parts: list[np.ndarray] = []

        # --- sim_state ---
        sim = obs["sim_state"]
        clock = float(sim["clock"])
        parts.append(np.array([clock], dtype=np.float32))

        for sid, cap in enumerate(capacities):
            st = sim["stations"][sid]
            parts.append(np.asarray(st["charger_status"], dtype=np.float32))

            qwt = [float(v) for v in st["queue_waiting_time"]][:mq]
            qd  = [float(v) for v in st["queue_demand"]][:mq]
            qwt += [0.0] * (mq - len(qwt))
            qd  += [0.0] * (mq - len(qd))
            parts.append(np.array(qwt, dtype=np.float32))
            parts.append(np.array(qd,  dtype=np.float32))

        m = sim["metrics"]
        parts.append(np.asarray(m["ev_served"],   dtype=np.float32)[:num_stations])
        parts.append(np.asarray(m["ev_queueing"], dtype=np.float32)[:num_stations])
        parts.append(np.asarray(m["queue_time"],  dtype=np.float32)[:num_stations])

        # --- commitment_features ---
        cf = obs["commitment_features"]
        parts.append(np.asarray(cf["commitment_count"],                dtype=np.float32))
        parts.append(np.asarray(cf["commitment_charge_demand"],        dtype=np.float32))
        parts.append(np.asarray(cf["earliest_expected_arrival_eta"],   dtype=np.float32))

        # --- current_ev ---
        ev = obs["current_ev"]
        parts.append(np.array([
            float(ev["station_id"]),
            float(ev["total_charge_demand"]),
        ], dtype=np.float32))
        ds_mask = np.zeros(num_stations, dtype=np.float32)
        for sid in ev["downstream_stations"]:
            if 0 <= int(sid) < num_stations:
                ds_mask[int(sid)] = 1.0
        parts.append(ds_mask)

        # --- future_demand ---
        parts.append(np.asarray(obs["future_demand"], dtype=np.float32)[:num_stations])

        return np.concatenate(parts)

    def action_masks(self) -> np.ndarray:
        """Expose the underlying env's action mask for BC collection and SB3."""
        return self.env.action_masks()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode_bank(data_dir: str) -> list[list]:
    """Return list-of-vehicle-lists from all CSV files under ``data_dir``."""
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV episode files found in '{data_dir}'")
    episodes = [load_demand_vehicles_from_csv(str(f)) for f in csv_files]
    print(f"  Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_charging_env(
    episode_bank: list[list],
    n_bins: int = 21,
    max_queue_len: int = 10,
    invalid_action_penalty: float = 0.0,
    seed: int = 0,
) -> Callable[[], gym.Env]:
    """Return a thunk that creates a single, fully-wrapped environment."""
    capacities = CAPACITY.tolist()
    tt_fn = travel_time_fn_from_matrix(TRAVEL_MATRIX)
    min_charge = float(MIN_SEG)

    def _init() -> gym.Env:
        env: gym.Env = EpisodeBankChargingEnv(
            episode_bank=episode_bank,
            station_capacities=capacities,
            travel_time_fn=tt_fn,
            min_first_charge=min_charge,
            min_second_charge=min_charge,
            n_bins=n_bins,
            invalid_action_penalty=invalid_action_penalty,
        )
        env = FlatObsWrapper(env, max_queue_len=max_queue_len)
        env = Monitor(env)
        env = ActionMasker(env, lambda wrapped_env: wrapped_env.get_wrapper_attr("action_masks")())
        env.reset(seed=seed)
        return env

    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    print("=== Maskable PPO (PopArt critic) — EV Charging ===")

    # Load episode data
    episode_bank = load_episode_bank(args.data_dir)

    # Vectorised training envs
    env_fns = [
        make_charging_env(
            episode_bank=episode_bank,
            n_bins=args.n_bins,
            max_queue_len=args.max_queue_len,
            invalid_action_penalty=args.invalid_action_penalty,
            seed=args.seed + i,
        )
        for i in range(args.n_envs)
    ]
    VecCls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    vec_env = VecCls(env_fns)

    # Eval env (no invalid-action penalty, fixed seed for reproducibility)
    eval_env = DummyVecEnv([
        make_charging_env(
            episode_bank=episode_bank,
            n_bins=args.n_bins,
            max_queue_len=args.max_queue_len,
            invalid_action_penalty=0.0,
            seed=args.seed + 9999,
        )
    ])

    # Directories
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)

    # Callbacks
    eval_freq = max(args.eval_freq // args.n_envs, 1)
    ckpt_freq = max(args.checkpoint_freq // args.n_envs, 1)

    from stable_baselines3.common.callbacks import CheckpointCallback

    callbacks = [
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
            name_prefix="maskable_ppo",
            verbose=1,
        ),
    ]

    # Model
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )
    model = PopArtMaskablePPO(
        policy=PopArtMaskablePolicy,
        env=vec_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        target_kl=args.target_kl,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        verbose=1,
        seed=args.seed,
    )

    # ---- Warm-start from BC pre-trained model ----
    if args.pretrained_path:
        print(f"  Loading BC pre-trained weights from {args.pretrained_path}")
        pretrained = MaskablePPO.load(args.pretrained_path, env=vec_env)
        src = pretrained.policy.state_dict()
        dst = model.policy.state_dict()

        loaded, skipped = [], []
        for key, param in src.items():
            if key in dst and dst[key].shape == param.shape:
                dst[key] = param
                loaded.append(key)
            else:
                skipped.append(key)

        # BC's value_net is nn.Linear (weight, bias); ours is PopArtHead
        # (linear.weight, linear.bias). Copy if shapes match.
        for suffix in ("weight", "bias"):
            bc_key = f"value_net.{suffix}"
            pop_key = f"value_net.linear.{suffix}"
            if bc_key in src and pop_key in dst and src[bc_key].shape == dst[pop_key].shape:
                dst[pop_key] = src[bc_key]
                loaded.append(f"{bc_key} -> {pop_key}")

        model.policy.load_state_dict(dst)
        print(f"  Loaded {len(loaded)} param tensors, skipped {len(skipped)}: {skipped}")
        del pretrained

    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.n
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  n_envs={args.n_envs}")
    print(f"  total_timesteps={args.total_timesteps:,}")
    print(f"  save_path={args.save_path}  log_dir={args.log_dir}")
    print(
        "  hyperparams="
        f"batch_size={args.batch_size}, "
        f"n_epochs={args.n_epochs}, "
        f"vf_coef={args.vf_coef}, "
        f"target_kl={args.target_kl}, "
        f"n_eval_episodes={args.n_eval_episodes}"
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(args.save_path, "maskable_ppo_final")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")

    vec_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Maskable PPO on the MultiStation EV Charging env"
    )

    # Data / env
    p.add_argument("--data_dir", type=str, default="data/train_dataset/normal",
                   help="Directory containing episode CSV files")
    p.add_argument("--n_bins", type=int, default=21,
                   help="Number of charge-split fraction bins")
    p.add_argument("--max_queue_len", type=int, default=10,
                   help="Max queue entries to encode per station in flat obs")
    p.add_argument("--invalid_action_penalty", type=float, default=0.0,
                   help="Extra penalty added to reward when an invalid action is taken")

    # Training
    p.add_argument("--total_timesteps", type=int, default=1_000_000)
    p.add_argument("--n_envs", type=int, default=4,
                   help="Number of parallel training environments")
    p.add_argument("--seed", type=int, default=42)

    # PPO hyperparameters
    p.add_argument("--learning_rate",   type=float, default=3e-4)
    p.add_argument("--n_steps",         type=int,   default=2048,
                   help="Rollout buffer steps per env before each PPO update")
    p.add_argument("--batch_size",      type=int,   default=64)
    p.add_argument("--n_epochs",        type=int,   default=10)
    p.add_argument("--gamma",           type=float, default=0.99)
    p.add_argument("--gae_lambda",      type=float, default=0.95)
    p.add_argument("--clip_range",      type=float, default=0.2)
    p.add_argument("--ent_coef",        type=float, default=0.01)
    p.add_argument("--vf_coef",         type=float, default=0.5)
    p.add_argument("--target_kl",       type=float, default=None,
                   help="Early stop PPO updates if approx KL exceeds 1.5 * target_kl")
    p.add_argument("--max_grad_norm",   type=float, default=0.5)

    # Warm-start
    p.add_argument("--pretrained_path", type=str, default=None,
                   help="Path to a BC pre-trained MaskablePPO .zip for warm-starting")

    # Eval / saving
    p.add_argument("--eval_freq",       type=int,   default=50_000,
                   help="Evaluate best model every N environment steps")
    p.add_argument("--n_eval_episodes", type=int,   default=10)
    p.add_argument("--checkpoint_freq", type=int,   default=100_000,
                   help="Save checkpoint every N environment steps")
    p.add_argument("--save_path",       type=str,   default="train/checkpoints")
    p.add_argument("--log_dir",         type=str,   default="train/logs")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
