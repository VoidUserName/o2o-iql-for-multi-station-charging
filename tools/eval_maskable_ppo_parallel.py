"""Parallel evaluation for MaskablePPO checkpoints on fixed episode CSVs."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.maskable.utils import get_action_masks

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulator.orchestrator import load_demand_vehicles_from_csv
from train.finetune.ppo_trainer import make_charging_env
from train.imitation.adaptive_bc import _load_pretrained_model_compat


def _load_episode_paths(data_dir: str) -> list[Path]:
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV episode files found in '{data_dir}'")
    return csv_files


def _evaluate_subset(
    args: tuple[list[str], str, int, int, int, str]
) -> dict:
    episode_paths, model_path, n_bins, max_queue_len, seed, label = args
    model = None
    rewards: list[float] = []
    lengths: list[int] = []
    mean_waiting_times: list[float] = []
    p95_waiting_times: list[float] = []
    max_waiting_times: list[float] = []
    load_imbalances: list[float] = []
    invalid_action_counts: list[int] = []
    decision_step_counts: list[int] = []
    episode_names: list[str] = []

    for idx, episode_path_str in enumerate(episode_paths):
        episode_path = Path(episode_path_str)
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

            obs = env.reset()
            done = False
            episode_reward = 0.0
            episode_length = 0
            episode_invalid_actions = 0
            final_info: dict = {}
            while not done:
                action_masks = get_action_masks(env)
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                    action_masks=action_masks,
                )
                obs, reward, dones, infos = env.step(action)
                episode_reward += float(reward[0])
                episode_length += 1
                episode_invalid_actions += int(bool(infos[0].get("invalid_action", False)))
                final_info = dict(infos[0])
                done = bool(dones[0])

            rewards.append(float(episode_reward))
            lengths.append(int(episode_length))
            mean_waiting_times.append(float(final_info.get("mean_waiting_time", 0.0)))
            p95_waiting_times.append(float(final_info.get("p95_waiting_time", 0.0)))
            max_waiting_times.append(float(final_info.get("max_waiting_time", 0.0)))
            load_imbalances.append(float(final_info.get("load_imbalance", 0.0)))
            invalid_action_counts.append(int(episode_invalid_actions))
            decision_step_counts.append(int(episode_length))
            episode_names.append(episode_path.name)
        finally:
            env.close()

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    mean_wait_arr = np.asarray(mean_waiting_times, dtype=np.float64)
    p95_wait_arr = np.asarray(p95_waiting_times, dtype=np.float64)
    max_wait_arr = np.asarray(max_waiting_times, dtype=np.float64)
    imbalance_arr = np.asarray(load_imbalances, dtype=np.float64)
    invalid_action_total = int(sum(invalid_action_counts))
    decision_step_total = int(sum(decision_step_counts))
    return {
        "label": label,
        "model_path": model_path,
        "n_eval_episodes": int(len(episode_paths)),
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std(ddof=0)),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "mean_ep_length": float(lengths_arr.mean()),
        "std_ep_length": float(lengths_arr.std(ddof=0)),
        "mean_waiting_time": float(mean_wait_arr.mean()),
        "std_waiting_time": float(mean_wait_arr.std(ddof=0)),
        "mean_p95_waiting_time": float(p95_wait_arr.mean()),
        "mean_max_waiting_time": float(max_wait_arr.mean()),
        "mean_load_imbalance": float(imbalance_arr.mean()),
        "invalid_action_rate": float(invalid_action_total / decision_step_total)
        if decision_step_total > 0
        else 0.0,
        "invalid_action_count": int(invalid_action_total),
        "decision_step_count": int(decision_step_total),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "episode_mean_waiting_times": mean_waiting_times,
        "episode_p95_waiting_times": p95_waiting_times,
        "episode_max_waiting_times": max_waiting_times,
        "episode_load_imbalances": load_imbalances,
        "episode_names": episode_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel evaluation for a MaskablePPO checkpoint"
    )
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--label", type=str, default="")
    parser.add_argument("--n_bins", type=int, default=21)
    parser.add_argument("--max_queue_len", type=int, default=10)
    parser.add_argument("--n_eval_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_paths = _load_episode_paths(args.data_dir)
    selected_paths = episode_paths[: min(args.n_eval_episodes, len(episode_paths))]
    if not selected_paths:
        raise ValueError("No evaluation episodes selected.")

    worker_count = max(1, int(args.num_workers))
    shard_size = int(np.ceil(len(selected_paths) / worker_count))
    shards = [
        [str(path) for path in selected_paths[i : i + shard_size]]
        for i in range(0, len(selected_paths), shard_size)
    ]
    print(f"Loaded {len(episode_paths)} episodes from {args.data_dir}")
    print(
        f"Evaluating {args.model} on the first {len(selected_paths)} fixed episode(s) "
        f"with {len(shards)} worker shard(s)"
    )

    task_args = [
        (shard, args.model, args.n_bins, args.max_queue_len, args.seed + idx * 10_000, args.label or Path(args.model).stem)
        for idx, shard in enumerate(shards)
    ]

    partials: list[dict] = []
    with ProcessPoolExecutor(max_workers=len(shards)) as executor:
        future_map = {executor.submit(_evaluate_subset, task): task for task in task_args}
        for future in as_completed(future_map):
            partials.append(future.result())

    partials.sort(key=lambda item: item["episode_names"][0])
    episode_rewards = [reward for part in partials for reward in part["episode_rewards"]]
    episode_lengths = [length for part in partials for length in part["episode_lengths"]]
    episode_mean_waiting_times = [
        value for part in partials for value in part["episode_mean_waiting_times"]
    ]
    episode_p95_waiting_times = [
        value for part in partials for value in part["episode_p95_waiting_times"]
    ]
    episode_max_waiting_times = [
        value for part in partials for value in part["episode_max_waiting_times"]
    ]
    episode_load_imbalances = [
        value for part in partials for value in part["episode_load_imbalances"]
    ]
    episode_names = [name for part in partials for name in part["episode_names"]]

    rewards_arr = np.asarray(episode_rewards, dtype=np.float64)
    lengths_arr = np.asarray(episode_lengths, dtype=np.int64)
    mean_wait_arr = np.asarray(episode_mean_waiting_times, dtype=np.float64)
    p95_wait_arr = np.asarray(episode_p95_waiting_times, dtype=np.float64)
    max_wait_arr = np.asarray(episode_max_waiting_times, dtype=np.float64)
    imbalance_arr = np.asarray(episode_load_imbalances, dtype=np.float64)
    invalid_action_total = int(sum(part["invalid_action_count"] for part in partials))
    decision_step_total = int(sum(part["decision_step_count"] for part in partials))
    summary = {
        "label": args.label or Path(args.model).stem,
        "model_path": args.model,
        "n_eval_episodes": int(len(selected_paths)),
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std(ddof=0)),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "mean_ep_length": float(lengths_arr.mean()),
        "std_ep_length": float(lengths_arr.std(ddof=0)),
        "mean_waiting_time": float(mean_wait_arr.mean()),
        "std_waiting_time": float(mean_wait_arr.std(ddof=0)),
        "mean_p95_waiting_time": float(p95_wait_arr.mean()),
        "mean_max_waiting_time": float(max_wait_arr.mean()),
        "mean_load_imbalance": float(imbalance_arr.mean()),
        "invalid_action_rate": float(invalid_action_total / decision_step_total)
        if decision_step_total > 0
        else 0.0,
        "invalid_action_count": int(invalid_action_total),
        "decision_step_count": int(decision_step_total),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_mean_waiting_times": episode_mean_waiting_times,
        "episode_p95_waiting_times": episode_p95_waiting_times,
        "episode_max_waiting_times": episode_max_waiting_times,
        "episode_load_imbalances": episode_load_imbalances,
        "episode_names": episode_names,
    }

    print(
        "mean_reward={mean_reward:.2f}  std_reward={std_reward:.2f}  "
        "mean_ep_length={mean_ep_length:.2f}".format(**summary)
    )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"results": [summary]}, indent=2), encoding="utf-8")
        print(f"Saved JSON summary to {output_path}")


if __name__ == "__main__":
    main()
