"""Evaluate one or more MaskablePPO checkpoints on a fixed dataset split.

Usage:
    python -m train.imitation.evaluate_policy \
        --data_dir data/train_dataset/bias \
        --n_eval_episodes 50 \
        --model logs/train_xxx/checkpoints/bc/bc_pretrained \
        --model logs/train_xxx/checkpoints/adaptive_bc/best_model \
        --model logs/train_xxx/checkpoints/adaptive_bc/adaptive_bc_final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from simulator.orchestrator import load_demand_vehicles_from_csv
from train.finetune.ppo_trainer import make_charging_env
from train.imitation.adaptive_bc import _load_pretrained_model_compat


def _load_episode_paths(data_dir: str) -> list[Path]:
    """Return all CSV episode paths under ``data_dir`` in sorted order."""
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV episode files found in '{data_dir}'")
    return csv_files


def _default_label(model_path: str) -> str:
    """Use the checkpoint stem as a human-readable label."""
    return Path(model_path).stem


def evaluate_checkpoint(
    *,
    model_path: str,
    label: str,
    episode_paths: list[Path],
    n_bins: int,
    max_queue_len: int,
    n_eval_episodes: int,
    seed: int,
) -> dict:
    """Evaluate a single checkpoint and return aggregate metrics.

    Each evaluation episode uses a dedicated one-episode environment so every
    checkpoint is scored on the exact same ordered subset of CSV files.
    """
    selected_paths = episode_paths[: min(n_eval_episodes, len(episode_paths))]
    if not selected_paths:
        raise ValueError("No evaluation episodes selected.")

    model = None
    rewards: list[float] = []
    lengths: list[int] = []
    episode_names: list[str] = []
    for idx, episode_path in enumerate(selected_paths):
        episode_bank = [[vehicle for vehicle in load_demand_vehicles_from_csv(str(episode_path))]]
        env = DummyVecEnv([
            make_charging_env(
                episode_bank=episode_bank,
                n_bins=n_bins,
                max_queue_len=max_queue_len,
                seed=seed + idx,
            )
        ])
        try:
            if model is None:
                model = _load_pretrained_model_compat(
                    model_path=model_path,
                    env=env,
                    tensorboard_log=None,
                    verbose=0,
                )
            else:
                model.set_env(env)

            reward_list, length_list = evaluate_policy(
                model,
                env,
                n_eval_episodes=1,
                deterministic=True,
                return_episode_rewards=True,
                warn=False,
                use_masking=True,
            )
            rewards.append(float(reward_list[0]))
            lengths.append(int(length_list[0]))
            episode_names.append(episode_path.name)
        finally:
            env.close()

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    return {
        "label": label,
        "model_path": model_path,
        "n_eval_episodes": int(len(selected_paths)),
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std(ddof=0)),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "mean_ep_length": float(lengths_arr.mean()),
        "std_ep_length": float(lengths_arr.std(ddof=0)),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "episode_names": episode_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more MaskablePPO checkpoints on a dataset split."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing episode CSVs to evaluate on",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        required=True,
        help="Checkpoint path (.zip optional). Repeat for multiple models.",
    )
    parser.add_argument(
        "--label",
        action="append",
        dest="labels",
        default=None,
        help="Optional label matching each --model entry in order.",
    )
    parser.add_argument("--n_bins", type=int, default=21)
    parser.add_argument("--max_queue_len", type=int, default=10)
    parser.add_argument("--n_eval_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save the evaluation summary as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.labels or []
    if labels and len(labels) != len(args.models):
        raise ValueError("Number of --label entries must match number of --model entries.")

    episode_paths = _load_episode_paths(args.data_dir)
    eval_count = min(args.n_eval_episodes, len(episode_paths))
    print(f"Loaded {len(episode_paths)} episodes from {args.data_dir}")
    print(f"Evaluating {len(args.models)} model(s) on the first {eval_count} fixed episode(s)")

    summaries = []
    for idx, model_path in enumerate(args.models):
        label = labels[idx] if labels else _default_label(model_path)
        print(f"\n=== Evaluating: {label} ===")
        summary = evaluate_checkpoint(
            model_path=model_path,
            label=label,
            episode_paths=episode_paths,
            n_bins=args.n_bins,
            max_queue_len=args.max_queue_len,
            n_eval_episodes=args.n_eval_episodes,
            seed=args.seed,
        )
        summaries.append(summary)
        print(
            "mean_reward={mean_reward:.2f}  std_reward={std_reward:.2f}  "
            "mean_ep_length={mean_ep_length:.2f}".format(**summary)
        )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps({"results": summaries}, indent=2),
            encoding="utf-8",
        )
        print(f"\nSaved JSON summary to {output_path}")


if __name__ == "__main__":
    main()
