"""Balanced dual-buffer offline-to-online IQL trainer."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

from train.iql.agent import DiscreteIQLAgent, IQLUpdateMetrics
from train.iql.data import (
    TransitionDataset,
    build_episode_bank_env,
    load_episode_bank_from_dir,
    load_offline_dataset,
)
from train.iql.iql_trainer import evaluate_agent
from train.o2o_iql.replay import BalancedReplayManager, PriorityRefreshStats


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_jsonl(path: str, payload: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


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


def run_offline_pretraining(
    agent: DiscreteIQLAgent,
    offline_dataset: TransitionDataset,
    args: argparse.Namespace,
    rng: np.random.Generator,
    metrics_path: str,
) -> None:
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


def _priority_stats_to_dict(stats: PriorityRefreshStats) -> dict[str, float | int | str]:
    return {
        "stage": "priority_refresh",
        "step": int(stats.step),
        "classifier_loss": float(stats.classifier_loss),
        "priority_entropy": float(stats.entropy),
        "priority_effective_sample_size": float(stats.effective_sample_size),
        "priority_top": float(stats.top_priority),
        "online_buffer_size": int(stats.online_buffer_size),
    }


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
    """Balanced dual-buffer online fine-tuning with prioritized offline replay."""
    env = build_episode_bank_env(
        episode_bank=train_episode_bank,
        n_bins=args.n_bins,
        max_queue_len=args.max_queue_len,
        invalid_action_penalty=args.invalid_action_penalty,
    )
    replay = BalancedReplayManager(
        offline_dataset=offline_dataset,
        obs_dim=offline_dataset.obs_dim,
        act_dim=offline_dataset.act_dim,
        online_buffer_size=args.online_buffer_size,
        online_sample_prob=args.online_sample_prob,
        min_online_samples=args.min_online_samples,
        priority_refresh_freq=args.priority_refresh_freq,
        priority_model_steps=args.priority_model_steps,
        priority_batch_size=args.priority_batch_size,
        priority_model_lr=args.priority_model_lr,
        priority_uniform_floor=args.priority_uniform_floor,
        priority_temperature=args.priority_temperature,
        priority_max_ratio=args.priority_max_ratio,
        device=args.device,
    )

    # Annealing schedules (PTGOOD §6.3: remove conservatism during online phase)
    offline_expectile = agent.expectile
    online_expectile = getattr(args, "online_expectile", 0.5)
    offline_temperature = agent.temperature
    online_temperature = getattr(args, "online_temperature", 1.0)
    raw_anneal = int(getattr(args, "anneal_steps", 0))
    anneal_steps = raw_anneal if raw_anneal > 0 else int(args.online_steps * 0.3)
    ucb_coef = float(getattr(args, "ucb_coef", 1.0))

    def _anneal(start: float, end: float, step: int) -> float:
        if anneal_steps <= 0:
            return end
        t = min(step / anneal_steps, 1.0)
        return start + (end - start) * t

    recent_returns: deque[float] = deque(maxlen=20)
    best_eval_reward = float("-inf")
    metric_window: list[IQLUpdateMetrics] = []
    source_counter: Counter[str] = Counter()

    try:
        obs, _ = env.reset(seed=args.seed)
        episode_return = 0.0
        episode_length = 0

        for step in range(1, args.online_steps + 1):
            # Anneal expectile and temperature
            agent.set_expectile(_anneal(offline_expectile, online_expectile, step))
            agent.set_temperature(_anneal(offline_temperature, online_temperature, step))

            action_mask = env.action_masks().astype(np.uint8)
            action = agent.act_ucb(obs, action_mask, ucb_coef=ucb_coef)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            replay.add_online_transition(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                done=done,
                action_mask=action_mask,
            )

            refresh_stats = replay.maybe_refresh_priorities(step=step, rng=rng)
            if refresh_stats is not None:
                payload = _priority_stats_to_dict(refresh_stats)
                print(
                    f"  Priority refresh step={refresh_stats.step:7d}"
                    f"  clf_loss={refresh_stats.classifier_loss:.4f}"
                    f"  ess={refresh_stats.effective_sample_size:.1f}"
                    f"  top_p={refresh_stats.top_priority:.6f}"
                )
                _append_jsonl(metrics_path, payload)

            episode_return += float(reward)
            episode_length += 1
            obs = next_obs

            if step >= args.start_training_after:
                for _ in range(args.updates_per_step):
                    batch, source = replay.sample(batch_size=args.batch_size, rng=rng)
                    metric_window.append(agent.update(batch))
                    source_counter[source] += 1

            if done:
                recent_returns.append(episode_return)
                print(
                    f"  Online episode finished"
                    f"  step={step:7d}/{args.online_steps}"
                    f"  return={episode_return:10.2f}"
                    f"  len={episode_length:4d}"
                    f"  online_buf={len(replay.online_buffer):6d}"
                )
                obs, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            if step % args.log_interval == 0 and metric_window:
                window_size = min(len(metric_window), args.log_interval * args.updates_per_step)
                averaged = _mean_metrics(metric_window[-window_size:])
                averaged.update(
                    {
                        "stage": "online",
                        "step": int(step),
                        "online_buffer_size": int(len(replay.online_buffer)),
                        "recent_return_mean": float(np.mean(recent_returns)) if recent_returns else 0.0,
                        "offline_updates": int(source_counter.get("offline", 0)),
                        "online_updates": int(source_counter.get("online", 0)),
                        "expectile": float(agent.expectile),
                        "temperature": float(agent.temperature),
                    }
                )
                print(
                    f"  Online step {step:7d}/{args.online_steps}"
                    f"  actor={averaged['actor_loss']:.4f}"
                    f"  critic={averaged['critic_loss']:.4f}"
                    f"  value={averaged['value_loss']:.4f}"
                    f"  recent_return={averaged['recent_return_mean']:.2f}"
                    f"  τ={agent.expectile:.3f} β={agent.temperature:.2f}"
                    f"  src=off:{averaged['offline_updates']} on:{averaged['online_updates']}"
                )
                _append_jsonl(metrics_path, averaged)
                source_counter.clear()

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
                    f"  Eval step={step:7d}"
                    f"  mean_reward={evaluation['mean_reward']:.2f}"
                    f"  std={evaluation['std_reward']:.2f}"
                )
                _append_jsonl(metrics_path, evaluation)
                if evaluation["mean_reward"] > best_eval_reward:
                    best_eval_reward = evaluation["mean_reward"]
                    best_path = os.path.join(save_path, "o2o_iql_best.pt")
                    agent.save(best_path, extra_state={"best_eval": evaluation, "step": step})
                    print(f"  Saved new best checkpoint -> {best_path}")

            if step % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(save_path, f"o2o_iql_step_{step}.pt")
                agent.save(checkpoint_path, extra_state={"step": step})
                print(f"  Saved checkpoint -> {checkpoint_path}")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced dual-buffer offline-to-online IQL trainer for EV charging"
    )

    parser.add_argument("--demand_dir", type=str, default="data/bc_dataset/demand")
    parser.add_argument("--solution_dir", type=str, default="data/bc_dataset/solutions")
    parser.add_argument("--train_data_dir", type=str, default="data/train_dataset/bias")
    parser.add_argument("--eval_data_dir", type=str, default="")

    parser.add_argument("--n_bins", type=int, default=21)
    parser.add_argument("--max_queue_len", type=int, default=10)
    parser.add_argument("--invalid_action_penalty", type=float, default=0.0)

    parser.add_argument("--offline_epochs", type=int, default=100)
    parser.add_argument("--online_steps", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--target_update_rate", type=float, default=5e-3)
    parser.add_argument("--exp_adv_max", type=float, default=100.0)
    parser.add_argument("--hidden_dim", type=int, default=256)

    parser.add_argument("--online_buffer_size", type=int, default=20_000)
    parser.add_argument("--online_sample_prob", type=float, default=0.6)
    parser.add_argument("--min_online_samples", type=int, default=2_000)
    parser.add_argument("--updates_per_step", type=int, default=1)
    parser.add_argument("--start_training_after", type=int, default=1)
    parser.add_argument("--exploration_epsilon", type=float, default=0.05,
                       help="(deprecated, kept for compat) ε-greedy not used with UCB")
    parser.add_argument("--ucb_coef", type=float, default=1.0,
                       help="UCB exploration coefficient λ: a = argmax [Q_mean + λ·Q_std]")
    parser.add_argument("--online_expectile", type=float, default=0.5,
                       help="Target expectile after annealing (0.5 = no conservatism)")
    parser.add_argument("--online_temperature", type=float, default=1.0,
                       help="Target AWR temperature after annealing")
    parser.add_argument("--anneal_steps", type=int, default=0,
                       help="Steps to anneal expectile/temperature (0 = 30%% of online_steps)")

    parser.add_argument("--priority_refresh_freq", type=int, default=5_000)
    parser.add_argument("--priority_model_steps", type=int, default=100)
    parser.add_argument("--priority_batch_size", type=int, default=512)
    parser.add_argument("--priority_model_lr", type=float, default=1e-3)
    parser.add_argument("--priority_uniform_floor", type=float, default=0.05)
    parser.add_argument("--priority_temperature", type=float, default=1.0)
    parser.add_argument("--priority_max_ratio", type=float, default=50.0)

    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--checkpoint_freq", type=int, default=50_000)
    parser.add_argument("--log_interval", type=int, default=10_000)

    parser.add_argument("--offline_limit_episodes", type=int, default=0)
    parser.add_argument("--train_limit_episodes", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--offline_dataset_cache", type=str, default="")
    parser.add_argument("--pretrained_checkpoint", type=str, default="")
    parser.add_argument("--save_path", type=str, default="train/o2o_iql/checkpoints")
    parser.add_argument("--log_dir", type=str, default="train/o2o_iql/logs")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.device = _resolve_device(args.device)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    _ensure_dir(args.save_path)
    _ensure_dir(args.log_dir)
    metrics_path = os.path.join(args.log_dir, "metrics.jsonl")

    print("=== O2O IQL with Balanced Dual Buffers ===")
    print(f"  device={args.device}")
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
        print(f"  Loading pretrained O2O-IQL checkpoint from '{args.pretrained_checkpoint}'")
        agent = DiscreteIQLAgent.load(args.pretrained_checkpoint, device=args.device)
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
            device=args.device,
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
        "o2o_iql_final.pt" if args.online_steps > 0 else "offline_iql_final.pt",
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
