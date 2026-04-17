"""Aggregate and compare all completed runs under runs/o2o_iql."""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.iql.agent import DiscreteIQLAgent
from train.iql.data import build_single_episode_env, load_episode_bank_from_dir


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _evaluate_checkpoint(task: tuple[str, str, str, int, int, float]) -> tuple[str, str, dict[str, Any]]:
    """Evaluate one checkpoint on one scenario."""
    scenario, seed_str, checkpoint_path_str, n_bins, max_queue_len, invalid_action_penalty = task
    checkpoint_path = Path(checkpoint_path_str)
    seed = int(seed_str)

    agent = DiscreteIQLAgent.load(checkpoint_path, device="cpu")
    episode_bank = load_episode_bank_from_dir(str(ROOT / "data" / "train_dataset" / scenario))

    rewards: list[float] = []
    lengths: list[int] = []
    waiting_times: list[float] = []
    p95_waiting_times: list[float] = []
    max_waiting_times: list[float] = []
    load_imbalances: list[float] = []
    invalid_action_total = 0
    decision_step_total = 0

    for episode_idx, vehicles in enumerate(episode_bank):
        env = build_single_episode_env(
            vehicles=vehicles,
            n_bins=n_bins,
            max_queue_len=max_queue_len,
            invalid_action_penalty=invalid_action_penalty,
        )
        try:
            obs, _ = env.reset(seed=seed + episode_idx)
            episode_reward = 0.0
            episode_length = 0
            episode_invalid_actions = 0
            final_info: dict[str, Any] = {}

            while True:
                action_mask = env.action_masks().astype(np.uint8)
                action = agent.act(obs, action_mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += float(reward)
                episode_length += 1
                episode_invalid_actions += int(bool(info.get("invalid_action", False)))
                final_info = dict(info)
                if terminated or truncated:
                    break
        finally:
            env.close()

        rewards.append(float(episode_reward))
        lengths.append(int(episode_length))
        waiting_times.append(float(final_info.get("mean_waiting_time", 0.0)))
        p95_waiting_times.append(float(final_info.get("p95_waiting_time", 0.0)))
        max_waiting_times.append(float(final_info.get("max_waiting_time", 0.0)))
        load_imbalances.append(float(final_info.get("load_imbalance", 0.0)))
        invalid_action_total += int(episode_invalid_actions)
        decision_step_total += int(episode_length)

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.float64)
    waiting_arr = np.asarray(waiting_times, dtype=np.float64)
    p95_arr = np.asarray(p95_waiting_times, dtype=np.float64)
    max_arr = np.asarray(max_waiting_times, dtype=np.float64)
    imbalance_arr = np.asarray(load_imbalances, dtype=np.float64)

    summary = {
        "scenario": scenario,
        "seed": seed,
        "checkpoint": str(checkpoint_path),
        "n_eval_episodes": int(len(episode_bank)),
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std(ddof=0)),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "mean_ep_length": float(lengths_arr.mean()),
        "std_ep_length": float(lengths_arr.std(ddof=0)),
        "mean_waiting_time": float(waiting_arr.mean()),
        "std_waiting_time": float(waiting_arr.std(ddof=0)),
        "mean_p95_waiting_time": float(p95_arr.mean()),
        "mean_max_waiting_time": float(max_arr.mean()),
        "mean_load_imbalance": float(imbalance_arr.mean()),
        "invalid_action_rate": float(invalid_action_total / decision_step_total)
        if decision_step_total > 0
        else 0.0,
        "invalid_action_count": int(invalid_action_total),
        "decision_step_count": int(decision_step_total),
    }
    return scenario, seed_str, summary


def _baseline_by_name(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {result["baseline_name"]: result for result in summary["results"]}


def _percent_improvement_higher_is_better(ours: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return float((ours - baseline) / abs(baseline) * 100.0)


def _percent_improvement_lower_is_better(ours: float, baseline: float) -> float:
    if baseline == 0.0:
        return 0.0
    return float((baseline - ours) / abs(baseline) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze all o2o_iql runs.")
    parser.add_argument("--run_root", type=str, default="runs/o2o_iql")
    parser.add_argument("--baseline_root", type=str, default="exps/results/baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--output_json", type=str, default="runs/o2o_iql/analysis_summary.json")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    baseline_root = Path(args.baseline_root)
    scenarios = [
        child.name
        for child in sorted(run_root.iterdir())
        if child.is_dir() and not child.name.startswith("_")
    ]

    tasks: list[tuple[str, str, str, int, int, float]] = []
    for scenario in scenarios:
        for seed_dir in sorted((run_root / scenario).glob("seed*")):
            if not seed_dir.is_dir():
                continue
            seed_str = seed_dir.name.replace("seed", "")
            checkpoint = seed_dir / "ckpt" / "o2o_iql_final.pt"
            if checkpoint.exists():
                tasks.append((scenario, seed_str, str(checkpoint), 21, 10, 0.0))

    if not tasks:
        raise FileNotFoundError(f"No checkpoints found under {run_root}")

    print(f"Evaluating {len(tasks)} checkpoints with max_workers={args.max_workers}")
    run_summaries: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=int(args.max_workers)) as executor:
        futures = {executor.submit(_evaluate_checkpoint, task): task for task in tasks}
        for future in as_completed(futures):
            scenario, seed_str, summary = future.result()
            run_summaries.append(summary)
            print(
                f"  done {scenario}/seed{seed_str}: "
                f"mean_reward={summary['mean_reward']:.2f}, "
                f"mean_waiting_time={summary['mean_waiting_time']:.4f}"
            )

    run_summaries.sort(key=lambda item: (item["scenario"], int(item["seed"])))

    baselines: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        baseline_path = baseline_root / scenario / "summary.json"
        if baseline_path.exists():
            baselines[scenario] = _baseline_by_name(_load_json(baseline_path))

    scenario_rows: dict[str, dict[str, Any]] = {}
    print("\n=== Scenario summary ===")
    for scenario in scenarios:
        scenario_runs = [item for item in run_summaries if item["scenario"] == scenario]
        if not scenario_runs:
            continue
        reward_mean = float(np.mean([item["mean_reward"] for item in scenario_runs]))
        reward_std = float(np.std([item["mean_reward"] for item in scenario_runs], ddof=0))
        wait_mean = float(np.mean([item["mean_waiting_time"] for item in scenario_runs]))
        wait_std = float(np.std([item["mean_waiting_time"] for item in scenario_runs], ddof=0))
        p95_mean = float(np.mean([item["mean_p95_waiting_time"] for item in scenario_runs]))
        max_wait_mean = float(np.mean([item["mean_max_waiting_time"] for item in scenario_runs]))

        scenario_rows[scenario] = {
            "mean_reward": reward_mean,
            "std_reward_across_seeds": reward_std,
            "mean_waiting_time": wait_mean,
            "std_waiting_time_across_seeds": wait_std,
            "mean_p95_waiting_time": p95_mean,
            "mean_max_waiting_time": max_wait_mean,
        }

        print(
            f"{scenario:8s}  reward={reward_mean:10.2f} +/- {reward_std:8.2f}  "
            f"wait={wait_mean:8.4f} +/- {wait_std:7.4f}  "
            f"p95={p95_mean:8.4f}  max={max_wait_mean:8.4f}"
        )

        if scenario in baselines:
            ref = baselines[scenario]
            reward_comparison = {}
            wait_comparison = {}
            for baseline_name in ("all-no-split", "greedy-downstream-min-queue", "greedy-no-split"):
                if baseline_name not in ref:
                    continue
                baseline = ref[baseline_name]
                reward_comparison[baseline_name] = _percent_improvement_higher_is_better(
                    reward_mean,
                    float(baseline["mean_reward"]),
                )
                wait_comparison[baseline_name] = _percent_improvement_lower_is_better(
                    wait_mean,
                    float(baseline["mean_waiting_time"]),
                )
            scenario_rows[scenario]["reward_improvement_vs_baselines_pct"] = reward_comparison
            scenario_rows[scenario]["wait_improvement_vs_baselines_pct"] = wait_comparison

    overall_reward_mean = float(np.mean([item["mean_reward"] for item in run_summaries]))
    overall_reward_std = float(np.std([item["mean_reward"] for item in run_summaries], ddof=0))
    overall_wait_mean = float(np.mean([item["mean_waiting_time"] for item in run_summaries]))
    overall_wait_std = float(np.std([item["mean_waiting_time"] for item in run_summaries], ddof=0))

    output = {
        "run_root": str(run_root),
        "baseline_root": str(baseline_root),
        "seed_for_eval": int(args.seed),
        "runs": run_summaries,
        "scenario_summary": scenario_rows,
        "overall_summary": {
            "mean_reward": overall_reward_mean,
            "std_reward_across_runs": overall_reward_std,
            "mean_waiting_time": overall_wait_mean,
            "std_waiting_time_across_runs": overall_wait_std,
        },
        "baselines": baselines,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved analysis to {output_path}")


if __name__ == "__main__":
    main()
