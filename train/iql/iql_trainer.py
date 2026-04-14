"""Offline-to-online IQL trainer for the EV charging environment."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch

from train.iql.agent import DiscreteIQLAgent, IQLUpdateMetrics
from train.iql.data import (
    TransitionDataset,
    build_episode_bank_env,
    build_single_episode_env,
    load_episode_bank_from_dir,
    load_offline_dataset,
)
from train.iql.replay import ReplayBuffer, sample_mixed_batch


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _mean_metrics(window: list[IQLUpdateMetrics]) -> dict[str, float]:
    if not window:
        return {}
    return {
        "actor_loss": float(np.mean([item.actor_loss for item in window])),
        "critic_loss": float(np.mean([item.critic_loss for item in window])),
        "value_loss": float(np.mean([item.value_loss for item in window])),
        "mean_advantage": float(np.mean([item.mean_advantage for item in window])),
        "mean_weight": float(np.mean([item.mean_weight for item in window])),
    }


def _append_jsonl(path: str, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def evaluate_agent(
    agent: DiscreteIQLAgent,
    episode_bank: list[list],
    n_eval_episodes: int,
    n_bins: int,
    max_queue_len: int,
    seed: int,
) -> dict[str, float]:
    """Run deterministic evaluation on the first N fixed episodes."""
    rewards: list[float] = []
    lengths: list[int] = []

    for episode_idx, vehicles in enumerate(episode_bank[: int(n_eval_episodes)]):
        env = build_single_episode_env(
            vehicles=vehicles,
            n_bins=n_bins,
            max_queue_len=max_queue_len,
        )
        try:
            obs, _ = env.reset(seed=seed + episode_idx)
            episode_reward = 0.0
            episode_length = 0

            while True:
                action_mask = env.action_masks().astype(np.uint8)
                action = agent.act(obs, action_mask, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                if terminated or truncated:
                    break
        finally:
            env.close()

        rewards.append(episode_reward)
        lengths.append(episode_length)

    return {
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "n_eval_episodes": int(len(rewards)),
    }


def run_offline_pretraining(
    agent: DiscreteIQLAgent,
    offline_dataset: TransitionDataset,
    args: argparse.Namespace,
    rng: np.random.Generator,
    metrics_path: str,
) -> None:
    """Train IQL only on the offline transition dataset."""
    updates_per_epoch = max(math.ceil(len(offline_dataset) / args.batch_size), 1)
    total_updates = int(args.offline_epochs) * updates_per_epoch
    history: list[IQLUpdateMetrics] = []

    print(
        f"  Offline dataset: {len(offline_dataset)} transitions"
        f"  obs_dim={offline_dataset.obs_dim}  act_dim={offline_dataset.act_dim}"
    )
    print(
        f"  Offline pretraining: {args.offline_epochs} epochs"
        f"  ({total_updates} gradient updates, batch_size={args.batch_size})"
    )

    for update_idx in range(1, total_updates + 1):
        batch = offline_dataset.sample(batch_size=args.batch_size, rng=rng)
        history.append(agent.update(batch))

        if update_idx % updates_per_epoch == 0:
            epoch = update_idx // updates_per_epoch
            averaged = _mean_metrics(history[-updates_per_epoch:])
            averaged["stage"] = "offline"
            averaged["epoch"] = int(epoch)
            averaged["update"] = int(update_idx)
            print(
                f"  Offline epoch {epoch:3d}/{args.offline_epochs}"
                f"  actor={averaged['actor_loss']:.4f}"
                f"  critic={averaged['critic_loss']:.4f}"
                f"  value={averaged['value_loss']:.4f}"
                f"  adv={averaged['mean_advantage']:.4f}"
            )
            _append_jsonl(metrics_path, averaged)


def run_online_finetuning(
    agent: DiscreteIQLAgent,
    offline_dataset: TransitionDataset,
    train_episode_bank: list[list],
    eval_episode_bank: list[list],
    args: argparse.Namespace,
    rng: np.random.Generator,
    metrics_path: str,
    save_path: str,
) -> None:
    """Continue training online with replay updates mixed with offline support."""
    env = build_episode_bank_env(
        episode_bank=train_episode_bank,
        n_bins=args.n_bins,
        max_queue_len=args.max_queue_len,
        invalid_action_penalty=args.invalid_action_penalty,
    )
    replay = ReplayBuffer(
        capacity=args.online_buffer_size,
        obs_dim=offline_dataset.obs_dim,
        act_dim=offline_dataset.act_dim,
    )
    recent_returns: deque[float] = deque(maxlen=20)
    best_eval_reward = float("-inf")
    metric_window: list[IQLUpdateMetrics] = []

    try:
        obs, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_length = 0

        for step in range(1, args.online_steps + 1):
            action_mask = env.action_masks().astype(np.uint8)
            if rng.random() < args.exploration_epsilon:
                valid_actions = np.flatnonzero(action_mask)
                action = int(rng.choice(valid_actions))
            else:
                action = agent.act(obs, action_mask, deterministic=False)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            replay.add(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
                action_mask=action_mask,
            )

            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if step >= args.start_training_after:
                for _ in range(args.updates_per_step):
                    batch = sample_mixed_batch(
                        offline_dataset=offline_dataset,
                        online_buffer=replay,
                        batch_size=args.batch_size,
                        offline_ratio=args.offline_mix_ratio,
                        rng=rng,
                    )
                    metric_window.append(agent.update(batch))

            if done:
                recent_returns.append(episode_return)
                print(
                    f"  Online episode finished"
                    f"  step={step:7d}/{args.online_steps}"
                    f"  return={episode_return:10.2f}"
                    f"  len={episode_length:4d}"
                    f"  replay={len(replay):6d}"
                )
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            if step % args.log_interval == 0 and metric_window:
                averaged = _mean_metrics(metric_window[-args.log_interval * args.updates_per_step :])
                averaged.update(
                    {
                        "stage": "online",
                        "step": int(step),
                        "replay_size": int(len(replay)),
                        "recent_return_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
                    }
                )
                print(
                    f"  Online step {step:7d}/{args.online_steps}"
                    f"  actor={averaged['actor_loss']:.4f}"
                    f"  critic={averaged['critic_loss']:.4f}"
                    f"  value={averaged['value_loss']:.4f}"
                    f"  recent_return={averaged['recent_return_mean']:.2f}"
                )
                _append_jsonl(metrics_path, averaged)

            if step % args.eval_freq == 0:
                evaluation = evaluate_agent(
                    agent=agent,
                    episode_bank=eval_episode_bank,
                    n_eval_episodes=args.n_eval_episodes,
                    n_bins=args.n_bins,
                    max_queue_len=args.max_queue_len,
                    seed=args.seed + 10_000 + step,
                )
                evaluation.update({"stage": "eval", "step": int(step)})
                print(
                    f"  Eval step={step:7d}  mean_reward={evaluation['mean_reward']:.2f}"
                    f"  std={evaluation['std_reward']:.2f}"
                )
                _append_jsonl(metrics_path, evaluation)
                if evaluation["mean_reward"] > best_eval_reward:
                    best_eval_reward = evaluation["mean_reward"]
                    best_path = os.path.join(save_path, "online_iql_best.pt")
                    agent.save(best_path, extra_state={"best_eval": evaluation, "step": step})
                    print(f"  Saved new best online checkpoint -> {best_path}")

            if step % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(save_path, f"online_iql_step_{step}.pt")
                agent.save(checkpoint_path, extra_state={"step": step})
                print(f"  Saved checkpoint -> {checkpoint_path}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline-to-online IQL trainer for the EV charging environment"
    )

    parser.add_argument("--demand_dir", type=str, default="data/bc_dataset/demand")
    parser.add_argument("--solution_dir", type=str, default="data/bc_dataset/solutions")
    parser.add_argument("--train_data_dir", type=str, default="data/train_dataset/bias")
    parser.add_argument("--eval_data_dir", type=str, default="")

    parser.add_argument("--n_bins", type=int, default=21)
    parser.add_argument("--max_queue_len", type=int, default=10)
    parser.add_argument("--invalid_action_penalty", type=float, default=0.0)

    parser.add_argument("--offline_epochs", type=int, default=100)
    parser.add_argument("--online_steps", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--target_update_rate", type=float, default=5e-3)
    parser.add_argument("--exp_adv_max", type=float, default=100.0)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--offline_mix_ratio", type=float, default=0.5)
    parser.add_argument("--online_buffer_size", type=int, default=100_000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--start_training_after", type=int, default=1)
    parser.add_argument("--exploration_epsilon", type=float, default=0.05)

    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--checkpoint_freq", type=int, default=25_000)
    parser.add_argument("--log_interval", type=int, default=1_000)

    parser.add_argument("--offline_limit_episodes", type=int, default=0)
    parser.add_argument("--train_limit_episodes", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--offline_dataset_cache", type=str, default="")
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--save_path", type=str, default="train/iql/checkpoints")
    parser.add_argument("--log_dir", type=str, default="train/iql/logs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _ensure_dir(args.save_path)
    _ensure_dir(args.log_dir)
    metrics_path = os.path.join(args.log_dir, "metrics.jsonl")

    print("=== IQL Offline-to-Online Training ===")
    print(f"  device={device}")
    print(f"  save_path={args.save_path}")
    print(f"  log_dir={args.log_dir}")

    if args.offline_dataset_cache and Path(args.offline_dataset_cache).exists():
        print(f"  Loading cached offline dataset from '{args.offline_dataset_cache}'")
        offline_dataset = TransitionDataset.load(args.offline_dataset_cache)
    else:
        offline_dataset = load_offline_dataset(
            demand_dir=args.demand_dir,
            solution_dir=args.solution_dir,
            n_bins=args.n_bins,
            max_queue_len=args.max_queue_len,
            seed=args.seed,
            limit_episodes=args.offline_limit_episodes,
        )
        if args.offline_dataset_cache:
            cache_path = Path(args.offline_dataset_cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            offline_dataset.save(cache_path)
            print(f"  Cached offline dataset -> {cache_path}")

    if args.pretrained_checkpoint:
        print(f"  Loading pretrained IQL checkpoint from '{args.pretrained_checkpoint}'")
        agent = DiscreteIQLAgent.load(args.pretrained_checkpoint, device=device)
    else:
        agent = DiscreteIQLAgent(
            obs_dim=offline_dataset.obs_dim,
            act_dim=offline_dataset.act_dim,
            hidden_dims=(args.hidden_dim, args.hidden_dim),
            learning_rate=args.learning_rate,
            discount=args.discount,
            expectile=args.expectile,
            temperature=args.temperature,
            target_update_rate=args.target_update_rate,
            exp_adv_max=args.exp_adv_max,
            device=device,
        )
        run_offline_pretraining(
            agent=agent,
            offline_dataset=offline_dataset,
            args=args,
            rng=rng,
            metrics_path=metrics_path,
        )
        offline_ckpt = os.path.join(args.save_path, "offline_iql.pt")
        agent.save(offline_ckpt, extra_state={"stage": "offline"})
        print(f"  Saved offline checkpoint -> {offline_ckpt}")

    eval_dir = args.eval_data_dir or args.train_data_dir
    eval_episode_bank = load_episode_bank_from_dir(
        eval_dir,
        limit=max(args.n_eval_episodes, 1),
    )
    offline_eval = evaluate_agent(
        agent=agent,
        episode_bank=eval_episode_bank,
        n_eval_episodes=args.n_eval_episodes,
        n_bins=args.n_bins,
        max_queue_len=args.max_queue_len,
        seed=args.seed + 5_000,
    )
    offline_eval.update({"stage": "offline_eval", "step": 0})
    print(
        f"  Offline eval mean_reward={offline_eval['mean_reward']:.2f}"
        f"  std={offline_eval['std_reward']:.2f}"
    )
    _append_jsonl(metrics_path, offline_eval)

    if args.online_steps > 0:
        train_episode_bank = load_episode_bank_from_dir(
            args.train_data_dir,
            limit=args.train_limit_episodes,
        )
        run_online_finetuning(
            agent=agent,
            offline_dataset=offline_dataset,
            train_episode_bank=train_episode_bank,
            eval_episode_bank=eval_episode_bank,
            args=args,
            rng=rng,
            metrics_path=metrics_path,
            save_path=args.save_path,
        )

    final_path = os.path.join(
        args.save_path,
        "online_iql_final.pt" if args.online_steps > 0 else "offline_iql_final.pt",
    )
    agent.save(
        final_path,
        extra_state={
            "stage": "online" if args.online_steps > 0 else "offline",
            "offline_eval": offline_eval,
        },
    )
    print(f"  Saved final checkpoint -> {final_path}")


if __name__ == "__main__":
    main()
