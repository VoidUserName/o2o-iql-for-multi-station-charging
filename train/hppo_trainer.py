"""Hierarchical Maskable-PPO trainer for the MultiStation EV Charging env.

Uses ``HierarchicalChargingEnv`` to decompose each charging decision into
two phases:

    Phase 0 (high-level):  split / no-split  →  Discrete(S + 1)
    Phase 1 (low-level) :  fraction-bin      →  Discrete(B)

Combined action space is Discrete(S + 1 + B) = Discrete(29) for 7 stations
and 21 bins — much smaller than the flat Discrete(168).

Usage:
    python -m train.hppo_trainer [--data_dir ...] [--total_timesteps ...]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from data.env_data import CAPACITY, MIN_SEG, TRAVEL_MATRIX
from envs.charging_env import EpisodeBankChargingEnv, travel_time_fn_from_matrix
from envs.hierarchy_env import HierarchicalChargingEnv
from simulator.orchestrator import load_demand_vehicles_from_csv


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_episode_bank(data_dir: str) -> list[list]:
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV episode files found in '{data_dir}'")
    episodes = [load_demand_vehicles_from_csv(str(f)) for f in csv_files]
    print(f"  Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_hierarchical_env(
    episode_bank: list[list],
    n_bins: int = 21,
    max_queue_len: int = 10,
    invalid_action_penalty: float = 0.0,
    seed: int = 0,
) -> Callable[[], gym.Env]:
    """Return a thunk that creates a single wrapped hierarchical env."""
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
        env = HierarchicalChargingEnv(env, max_queue_len=max_queue_len)
        env = Monitor(env)
        env = ActionMasker(env, lambda e: e.get_wrapper_attr("action_masks")())
        env.reset(seed=seed)
        return env

    return _init


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    print("=== Hierarchical Maskable PPO — EV Charging ===")

    episode_bank = load_episode_bank(args.data_dir)

    # Vectorised training envs
    env_fns = [
        make_hierarchical_env(
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

    # Eval env
    eval_env = DummyVecEnv([
        make_hierarchical_env(
            episode_bank=episode_bank,
            n_bins=args.n_bins,
            max_queue_len=args.max_queue_len,
            invalid_action_penalty=0.0,
            seed=args.seed + 9999,
        )
    ])

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir,   exist_ok=True)

    eval_freq = max(args.eval_freq // args.n_envs, 1)
    ckpt_freq = max(args.checkpoint_freq // args.n_envs, 1)

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
            name_prefix="hppo",
            verbose=1,
        ),
    ]

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    model = MaskablePPO(
        policy="MlpPolicy",
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
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        verbose=1,
        seed=args.seed,
    )

    obs_dim = vec_env.observation_space.shape[0]
    act_dim = vec_env.action_space.n
    print(f"  obs_dim={obs_dim}  act_dim={act_dim}  (flat was ~498 / 168)")
    print(f"  n_envs={args.n_envs}  total_timesteps={args.total_timesteps:,}")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    final_path = os.path.join(args.save_path, "hppo_final")
    model.save(final_path)
    print(f"\nFinal model saved → {final_path}.zip")

    vec_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Hierarchical Maskable PPO on EV Charging"
    )

    p.add_argument("--data_dir", type=str, default="data/train_dataset/normal")
    p.add_argument("--n_bins",   type=int, default=21)
    p.add_argument("--max_queue_len", type=int, default=10)
    p.add_argument("--invalid_action_penalty", type=float, default=0.0)

    p.add_argument("--total_timesteps", type=int,   default=1_000_000)
    p.add_argument("--n_envs",          type=int,   default=4)
    p.add_argument("--seed",            type=int,   default=42)

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

    p.add_argument("--eval_freq",       type=int,   default=50_000)
    p.add_argument("--n_eval_episodes", type=int,   default=10)
    p.add_argument("--checkpoint_freq", type=int,   default=100_000)
    p.add_argument("--save_path",       type=str,   default="train/checkpoints/hppo")
    p.add_argument("--log_dir",         type=str,   default="train/logs/hppo")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
