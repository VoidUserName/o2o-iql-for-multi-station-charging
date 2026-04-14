"""Evaluate baseline policies on the EV charging environment.

Currently implemented baselines:
    - all-no-split: always choose the canonical no-split action
    - greedy-downstream-min-queue: choose the downstream station with the
      shortest queue, then use a valid split action close to a balanced split
    - greedy-no-split: choose among the current station and downstream stations
      by shortest queue, then by nearest travel time; if the current station
      wins, take no-split, otherwise take a valid split toward the selected
      station

Usage:
    python -m train.evalution \
        --data_dir data/train_dataset/bias \
        --n_eval_episodes 50
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable

import numpy as np

from data.env_data import CAPACITY, MIN_SEG, TRAVEL_MATRIX
from envs.charging_env import EpisodeBankChargingEnv, travel_time_fn_from_matrix
from envs.maskable_actions import (
    decode_maskable_action,
    iter_valid_maskable_actions,
    no_split_action_int,
)
from simulator.orchestrator import load_demand_vehicles_from_csv


BaselineFn = Callable[[EpisodeBankChargingEnv], int]


def _load_episode_paths(data_dir: str) -> list[Path]:
    """Return all CSV episode paths under ``data_dir`` in sorted order."""
    csv_files = sorted(Path(data_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV episode files found in '{data_dir}'")
    return csv_files


def _load_dataset_parts(data_dir: str) -> list[tuple[str, list[Path]]]:
    """Return one or more dataset parts found under ``data_dir``.

    If ``data_dir`` contains CSV files directly, it is treated as a single
    evaluation split. If it contains subdirectories, each subdirectory is
    treated as its own split and must contain CSV episode files.
    """
    root = Path(data_dir)
    direct_csvs = sorted(root.glob("*.csv"))
    if direct_csvs:
        return [(root.name, direct_csvs)]

    split_dirs = sorted(child for child in root.iterdir() if child.is_dir())
    if not split_dirs:
        raise FileNotFoundError(
            f"No CSV episode files or split directories found in '{data_dir}'"
        )

    dataset_parts: list[tuple[str, list[Path]]] = []
    for split_dir in split_dirs:
        csv_files = sorted(split_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV episode files found in split directory '{split_dir}'"
            )
        dataset_parts.append((split_dir.name, csv_files))
    return dataset_parts


def _load_single_episode_env(episode_path: Path, n_bins: int) -> EpisodeBankChargingEnv:
    """Create a one-episode environment for a single CSV file."""
    episode_bank = [[vehicle for vehicle in load_demand_vehicles_from_csv(str(episode_path))]]
    return EpisodeBankChargingEnv(
        episode_bank=episode_bank,
        station_capacities=CAPACITY.tolist(),
        travel_time_fn=travel_time_fn_from_matrix(TRAVEL_MATRIX),
        min_first_charge=float(MIN_SEG),
        min_second_charge=float(MIN_SEG),
        n_bins=int(n_bins),
    )


def _all_no_split_action(env: EpisodeBankChargingEnv) -> int:
    """Baseline that always chooses the canonical no-split action."""
    return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)


def _greedy_downstream_min_queue_action(env: EpisodeBankChargingEnv) -> int:
    """Choose the downstream station with the shortest current queue.

    The heuristic is:
    1. Look only at downstream stations for the pending vehicle.
    2. Prefer the station with the smallest queue length.
    3. Break ties by smaller total queued waiting time, then route order.
    4. For that station, pick a valid split action closest to a balanced split.
    5. Fall back to no-split if no valid split exists.
    """
    if env.pending_vehicle is None:
        return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)

    downstream_stations = [
        int(station_id)
        for station_id in env.pending_vehicle.route[1:]
        if 0 <= int(station_id) < env.num_stations
    ]
    if not downstream_stations:
        return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)

    observation = env._current_observation()  # noqa: SLF001 - evaluation helper
    station_payloads = observation["sim_state"]["stations"]

    valid_split_actions: dict[int, list[tuple[int, int]]] = {}
    for action in iter_valid_maskable_actions(
        route=env.pending_vehicle.route,
        n_bins=env.n_bins,
        total_duration=float(env.pending_vehicle.duration),
        t_first_min=float(env.min_first_charge),
        t_second_min=float(env.min_second_charge),
        num_stations=env.num_stations,
    ):
        second_choice, frac_bin = decode_maskable_action(
            action_int=action,
            n_bins=env.n_bins,
            num_stations=env.num_stations,
        )
        if second_choice == env.num_stations:
            continue
        valid_split_actions.setdefault(int(second_choice), []).append((int(action), int(frac_bin)))

    if not valid_split_actions:
        return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)

    route_order = {station_id: idx for idx, station_id in enumerate(downstream_stations)}

    def _queue_score(station_id: int) -> tuple[int, float, int]:
        queue_waiting_time = station_payloads[station_id]["queue_waiting_time"]
        return (
            len(queue_waiting_time),
            float(sum(float(value) for value in queue_waiting_time)),
            int(route_order.get(station_id, len(route_order))),
        )

    chosen_station = min(valid_split_actions, key=_queue_score)
    split_candidates = valid_split_actions[chosen_station]
    target_frac_bin = (env.n_bins - 1) / 2.0
    chosen_action = min(
        split_candidates,
        key=lambda item: (abs(item[1] - target_frac_bin), item[1]),
    )[0]
    return int(chosen_action)


def _greedy_no_split_action(env: EpisodeBankChargingEnv) -> int:
    """Choose the station with the smallest queue, preferring the nearest one.

    Candidate stations include the current station itself and all downstream
    stations on the route. If the current station is selected, the action is
    the canonical no-split action. Otherwise, we select a valid split action
    that targets the chosen downstream station.
    """
    if env.pending_vehicle is None:
        return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)

    route = [
        int(station_id)
        for station_id in env.pending_vehicle.route
        if 0 <= int(station_id) < env.num_stations
    ]
    if not route:
        return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)

    current_station = int(route[0])
    candidate_stations = []
    seen: set[int] = set()
    for station_id in route:
        if station_id in seen:
            continue
        seen.add(station_id)
        candidate_stations.append(int(station_id))

    observation = env._current_observation()  # noqa: SLF001 - evaluation helper
    station_payloads = observation["sim_state"]["stations"]

    def _candidate_score(station_id: int) -> tuple[int, float, int]:
        queue_waiting_time = station_payloads[station_id]["queue_waiting_time"]
        queue_len = len(queue_waiting_time)
        travel_time = float(TRAVEL_MATRIX[current_station][station_id])
        return (queue_len, travel_time, int(station_id))

    chosen_station = min(candidate_stations, key=_candidate_score)
    if chosen_station == current_station:
        return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)

    valid_split_actions: list[tuple[int, int]] = []
    for action in iter_valid_maskable_actions(
        route=env.pending_vehicle.route,
        n_bins=env.n_bins,
        total_duration=float(env.pending_vehicle.duration),
        t_first_min=float(env.min_first_charge),
        t_second_min=float(env.min_second_charge),
        num_stations=env.num_stations,
    ):
        second_choice, frac_bin = decode_maskable_action(
            action_int=action,
            n_bins=env.n_bins,
            num_stations=env.num_stations,
        )
        if second_choice == int(chosen_station):
            valid_split_actions.append((int(action), int(frac_bin)))

    if not valid_split_actions:
        return no_split_action_int(n_bins=env.n_bins, num_stations=env.num_stations)

    target_frac_bin = (env.n_bins - 1) / 2.0
    chosen_action = min(
        valid_split_actions,
        key=lambda item: (abs(item[1] - target_frac_bin), item[1]),
    )[0]
    return int(chosen_action)


BASELINES: dict[str, BaselineFn] = {
    "all-no-split": _all_no_split_action,
    "greedy-downstream-min-queue": _greedy_downstream_min_queue_action,
    "greedy-no-split": _greedy_no_split_action,
}


def _evaluate_single_episode(
    args: tuple[int, str, str, int, int],
) -> dict:
    """Evaluate one episode for one baseline in a worker process."""
    episode_idx, episode_path_str, baseline_name, n_bins, seed = args
    episode_path = Path(episode_path_str)
    baseline_fn = BASELINES[baseline_name]
    env = _load_single_episode_env(episode_path, n_bins=n_bins)
    try:
        obs, info = env.reset(seed=seed + episode_idx)
        del obs  # The baseline does not need the observation.

        episode_reward = 0.0
        episode_length = 0
        final_info = dict(info)
        episode_invalid_actions = 0

        while env.pending_vehicle is not None:
            action = int(baseline_fn(env))
            _, reward, terminated, truncated, step_info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            episode_invalid_actions += int(bool(step_info.get("invalid_action", False)))
            final_info = dict(step_info)
            if terminated or truncated:
                break

        return {
            "episode_reward": float(episode_reward),
            "episode_length": int(episode_length),
            "episode_name": episode_path.name,
            "mean_waiting_time": float(final_info.get("mean_waiting_time", 0.0)),
            "p95_waiting_time": float(final_info.get("p95_waiting_time", 0.0)),
            "max_waiting_time": float(final_info.get("max_waiting_time", 0.0)),
            "load_imbalance": float(final_info.get("load_imbalance", 0.0)),
            "invalid_action_count": int(episode_invalid_actions),
            "decision_step_count": int(episode_length),
        }
    finally:
        env.close()


def _evaluate_split_baseline_task(
    args: tuple[str, str, str, int, int],
) -> tuple[str, str, dict]:
    """Evaluate one baseline on one split in a worker process."""
    split_name, split_dir_str, baseline_name, n_bins, seed = args
    episode_paths = _load_episode_paths(split_dir_str)
    summary = evaluate_baseline(
        baseline_name=baseline_name,
        baseline_fn=BASELINES[baseline_name],
        episode_paths=episode_paths,
        n_bins=n_bins,
        n_eval_episodes=0,
        seed=seed,
        num_workers=1,
    )
    return split_name, baseline_name, summary


def evaluate_baseline(
    *,
    baseline_name: str,
    baseline_fn: BaselineFn,
    episode_paths: list[Path],
    n_bins: int,
    n_eval_episodes: int,
    seed: int,
    num_workers: int = 1,
) -> dict:
    """Evaluate a baseline on a fixed ordered subset of episode CSVs."""
    if n_eval_episodes <= 0:
        selected_paths = episode_paths
    else:
        selected_paths = episode_paths[: min(n_eval_episodes, len(episode_paths))]
    if not selected_paths:
        raise ValueError("No evaluation episodes selected.")

    worker_args = [
        (idx, str(episode_path), baseline_name, n_bins, seed)
        for idx, episode_path in enumerate(selected_paths)
    ]
    if num_workers <= 1:
        records = [_evaluate_single_episode(item) for item in worker_args]
    else:
        with ProcessPoolExecutor(max_workers=int(num_workers)) as executor:
            records = list(executor.map(_evaluate_single_episode, worker_args, chunksize=1))

    rewards = [float(record["episode_reward"]) for record in records]
    lengths = [int(record["episode_length"]) for record in records]
    episode_names = [str(record["episode_name"]) for record in records]
    mean_waiting_times = [float(record["mean_waiting_time"]) for record in records]
    p95_waiting_times = [float(record["p95_waiting_time"]) for record in records]
    max_waiting_times = [float(record["max_waiting_time"]) for record in records]
    load_imbalances = [float(record["load_imbalance"]) for record in records]
    invalid_action_total = int(sum(int(record["invalid_action_count"]) for record in records))
    decision_step_total = int(sum(int(record["decision_step_count"]) for record in records))

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    mean_wait_arr = np.asarray(mean_waiting_times, dtype=np.float64)
    p95_wait_arr = np.asarray(p95_waiting_times, dtype=np.float64)
    max_wait_arr = np.asarray(max_waiting_times, dtype=np.float64)
    imbalance_arr = np.asarray(load_imbalances, dtype=np.float64)

    return {
        "label": baseline_name,
        "baseline_name": baseline_name,
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
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "episode_names": episode_names,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline policies on the EV charging environment."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory containing episode CSVs to evaluate on",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        dest="baselines",
        default=None,
        help="Baseline name to evaluate. Repeat for multiple baselines.",
    )
    parser.add_argument("--n_bins", type=int, default=21)
    parser.add_argument("--n_eval_episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes used to evaluate episodes in parallel. "
        "Use 0 for auto (parallel in batch mode, single-process otherwise).",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="",
        help="Optional path to save the evaluation summary as JSON.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="exps/results/baselines",
        help="Root directory for batch baseline outputs when data_dir contains splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_names = args.baselines or ["all-no-split"]
    unknown = [name for name in baseline_names if name not in BASELINES]
    if unknown:
        raise ValueError(
            f"Unknown baseline(s): {', '.join(unknown)}. "
            f"Available baselines: {', '.join(sorted(BASELINES))}"
        )

    root_path = Path(args.data_dir)
    direct_csvs = sorted(root_path.glob("*.csv"))
    dataset_parts = _load_dataset_parts(args.data_dir)
    is_batch_mode = not bool(direct_csvs)
    num_workers = int(args.num_workers)
    if num_workers <= 0:
        num_workers = min(8, (os.cpu_count() or 1)) if is_batch_mode else 1

    if is_batch_mode:
        output_root = Path(args.output_root)
        total_episodes = sum(len(paths) for _, paths in dataset_parts)
        print(f"Loaded {total_episodes} episodes across {len(dataset_parts)} split(s) from {args.data_dir}")
        print(
            f"Evaluating {len(baseline_names)} baseline(s) on all episodes in each split"
        )
        batch_workers = int(num_workers)
        task_args = [
            (
                split_name,
                str(Path(args.data_dir) / split_name),
                baseline_name,
                args.n_bins,
                args.seed,
            )
            for split_name, _ in dataset_parts
            for baseline_name in baseline_names
        ]
        split_summaries: dict[str, dict[str, dict]] = {
            split_name: {} for split_name, _ in dataset_parts
        }
        split_output_dirs = {
            split_name: output_root / split_name for split_name, _ in dataset_parts
        }
        for split_output_dir in split_output_dirs.values():
            split_output_dir.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(max_workers=batch_workers) as executor:
            future_map = {
                executor.submit(_evaluate_split_baseline_task, task): task
                for task in task_args
            }
            for future in as_completed(future_map):
                split_name, baseline_name, summary = future.result()
                split_summaries[split_name][baseline_name] = summary
                baseline_output_path = split_output_dirs[split_name] / f"{baseline_name}.json"
                baseline_output_path.write_text(
                    json.dumps(
                        {
                            "data_dir": str(Path(args.data_dir) / split_name),
                            "split_name": split_name,
                            "results": [summary],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                print(
                    f"Saved {split_name}/{baseline_name} -> {baseline_output_path}\n"
                    "mean_reward={mean_reward:.2f}  std_reward={std_reward:.2f}  "
                    "mean_ep_length={mean_ep_length:.2f}  mean_waiting_time={mean_waiting_time:.2f}".format(
                        **summary
                    )
                )

        for split_name, _ in dataset_parts:
            summary_output_path = split_output_dirs[split_name] / "summary.json"
            ordered_results = [
                split_summaries[split_name][baseline_name] for baseline_name in baseline_names
            ]
            summary_output_path.write_text(
                json.dumps(
                    {
                        "data_dir": str(Path(args.data_dir) / split_name),
                        "split_name": split_name,
                        "results": ordered_results,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved split summary to {summary_output_path}")
    else:
        episode_paths = dataset_parts[0][1]
        eval_count = min(args.n_eval_episodes, len(episode_paths))
        print(f"Loaded {len(episode_paths)} episodes from {args.data_dir}")
        print(
            f"Evaluating {len(baseline_names)} baseline(s) on the first {eval_count} fixed episode(s)"
        )

        summaries: list[dict] = []
        for baseline_name in baseline_names:
            print(f"\n=== Evaluating baseline: {baseline_name} ===")
            summary = evaluate_baseline(
                baseline_name=baseline_name,
                baseline_fn=BASELINES[baseline_name],
                episode_paths=episode_paths,
                n_bins=args.n_bins,
                n_eval_episodes=args.n_eval_episodes,
                seed=args.seed,
                num_workers=num_workers,
            )
            summaries.append(summary)
            print(
                "mean_reward={mean_reward:.2f}  std_reward={std_reward:.2f}  "
                "mean_ep_length={mean_ep_length:.2f}  mean_waiting_time={mean_waiting_time:.2f}".format(
                    **summary
                )
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
