"""Evaluate trained ablation checkpoints on all scenario splits.

For every (variant, seed) that produced a checkpoint, roll out the agent
deterministically on every episode under data/train_dataset/<split>/ for each
split in configs.SPLITS. The per-split JSON schema matches
exps/results/baselines/<split>/<label>.json so results plug into the same
downstream plotting/analysis.

Usage:
    python -m exps.ablations.evaluate_ablation
    python -m exps.ablations.evaluate_ablation --variants full_o2o_iql no_ucb
    python -m exps.ablations.evaluate_ablation --seeds 42
    python -m exps.ablations.evaluate_ablation --n_eval_episodes 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from exps.ablations import configs
from train.iql.agent import DiscreteIQLAgent
from train.iql.data import build_single_episode_env
from simulator.orchestrator import load_demand_vehicles_from_csv


def _discover_ckpt(variant: str, seed: int) -> Path | None:
    save_dir = Path(configs.variant_save_path(variant, seed))
    best = save_dir / "o2o_iql_best.pt"
    if best.exists():
        return best
    final = save_dir / "o2o_iql_final.pt"
    if final.exists():
        return final
    offline_final = save_dir / "offline_iql_final.pt"
    if offline_final.exists():
        return offline_final
    return None


def _evaluate_agent_on_split(
    agent: DiscreteIQLAgent,
    split_dir: Path,
    n_bins: int,
    max_queue_len: int,
    seed: int,
    n_eval_episodes: int,
) -> dict:
    csv_files = sorted(split_dir.glob("*.csv"))
    if n_eval_episodes > 0:
        csv_files = csv_files[: int(n_eval_episodes)]
    if not csv_files:
        raise FileNotFoundError(f"No CSV episodes under '{split_dir}'")

    rewards: list[float] = []
    lengths: list[int] = []
    episode_names: list[str] = []
    mean_waits: list[float] = []
    p95_waits: list[float] = []
    max_waits: list[float] = []
    imbalances: list[float] = []
    invalid_count = 0
    decision_count = 0

    for idx, csv_path in enumerate(csv_files):
        vehicles = load_demand_vehicles_from_csv(str(csv_path))
        env = build_single_episode_env(
            vehicles=vehicles,
            n_bins=n_bins,
            max_queue_len=max_queue_len,
        )
        try:
            obs, info = env.reset(seed=seed + idx)
            ep_reward = 0.0
            ep_length = 0
            ep_invalid = 0
            final_info = dict(info)
            while True:
                mask = env.action_masks().astype(np.uint8)
                action = agent.act(obs, mask, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                ep_reward += float(reward)
                ep_length += 1
                if bool(step_info.get("invalid_action", False)):
                    ep_invalid += 1
                final_info = dict(step_info)
                if terminated or truncated:
                    break
        finally:
            env.close()

        rewards.append(ep_reward)
        lengths.append(ep_length)
        episode_names.append(csv_path.name)
        mean_waits.append(float(final_info.get("mean_waiting_time", 0.0)))
        p95_waits.append(float(final_info.get("p95_waiting_time", 0.0)))
        max_waits.append(float(final_info.get("max_waiting_time", 0.0)))
        imbalances.append(float(final_info.get("load_imbalance", 0.0)))
        invalid_count += int(ep_invalid)
        decision_count += int(ep_length)

    rewards_arr = np.asarray(rewards, dtype=np.float64)
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    return {
        "n_eval_episodes": int(len(rewards)),
        "mean_reward": float(rewards_arr.mean()),
        "std_reward": float(rewards_arr.std(ddof=0)),
        "min_reward": float(rewards_arr.min()),
        "max_reward": float(rewards_arr.max()),
        "mean_ep_length": float(lengths_arr.mean()),
        "std_ep_length": float(lengths_arr.std(ddof=0)),
        "mean_waiting_time": float(np.mean(mean_waits)),
        "std_waiting_time": float(np.std(mean_waits, ddof=0)),
        "mean_p95_waiting_time": float(np.mean(p95_waits)),
        "mean_max_waiting_time": float(np.mean(max_waits)),
        "mean_load_imbalance": float(np.mean(imbalances)),
        "invalid_action_rate": float(invalid_count / decision_count) if decision_count > 0 else 0.0,
        "invalid_action_count": int(invalid_count),
        "decision_step_count": int(decision_count),
        "episode_rewards": rewards,
        "episode_lengths": lengths,
        "episode_names": episode_names,
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _aggregate(per_seed: list[dict]) -> dict:
    if not per_seed:
        return {}
    mean_rewards = [r["mean_reward"] for r in per_seed]
    return {
        "n_seeds": len(per_seed),
        "mean_reward_mean": float(np.mean(mean_rewards)),
        "mean_reward_std": float(np.std(mean_rewards, ddof=0)),
        "mean_reward_min": float(np.min(mean_rewards)),
        "mean_reward_max": float(np.max(mean_rewards)),
        "mean_ep_length_mean": float(np.mean([r["mean_ep_length"] for r in per_seed])),
        "mean_waiting_time_mean": float(np.mean([r["mean_waiting_time"] for r in per_seed])),
        "mean_p95_waiting_time_mean": float(np.mean([r["mean_p95_waiting_time"] for r in per_seed])),
        "mean_max_waiting_time_mean": float(np.mean([r["mean_max_waiting_time"] for r in per_seed])),
        "mean_load_imbalance_mean": float(np.mean([r["mean_load_imbalance"] for r in per_seed])),
        "invalid_action_rate_mean": float(np.mean([r["invalid_action_rate"] for r in per_seed])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate O2O IQL ablation checkpoints.")
    parser.add_argument("--variants", nargs="*", default=None, choices=list(configs.ABLATIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=None, choices=configs.SPLITS)
    parser.add_argument("--n_bins", type=int, default=21)
    parser.add_argument("--max_queue_len", type=int, default=10)
    parser.add_argument("--n_eval_episodes", type=int, default=0,
                        help="0 = all episodes in split (matches baselines/summary.json)")
    parser.add_argument("--seed", type=int, default=42, help="Env reset seed base")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    variants = args.variants or list(configs.ABLATIONS)
    seeds = args.seeds or list(configs.SEEDS)
    splits = args.splits or list(configs.SPLITS)

    overall: dict[str, dict[str, dict]] = {}

    for variant in variants:
        overall[variant] = {}
        for split in splits:
            split_dir = Path(configs.EVAL_DATA_ROOT) / split
            per_seed_records: list[dict] = []
            for seed in seeds:
                ckpt = _discover_ckpt(variant, seed)
                if ckpt is None:
                    print(f"  [skip] {variant}/seed{seed}: no checkpoint found")
                    continue
                print(f"  Loading {ckpt}")
                agent = DiscreteIQLAgent.load(str(ckpt), device=args.device)
                print(f"  Evaluating {variant}/seed{seed} on split={split}")
                result = _evaluate_agent_on_split(
                    agent=agent,
                    split_dir=split_dir,
                    n_bins=args.n_bins,
                    max_queue_len=args.max_queue_len,
                    seed=args.seed,
                    n_eval_episodes=args.n_eval_episodes,
                )
                result["label"] = f"{variant}/seed{seed}"
                result["variant"] = variant
                result["seed"] = int(seed)
                result["split_name"] = split
                result["data_dir"] = str(split_dir)
                result["checkpoint"] = str(ckpt)

                out_path = Path(configs.variant_result_dir(variant)) / split / f"seed{seed}.json"
                _write_json(out_path, {
                    "data_dir": str(split_dir),
                    "split_name": split,
                    "variant": variant,
                    "seed": int(seed),
                    "results": [result],
                })
                print(
                    f"    mean_reward={result['mean_reward']:.2f}"
                    f"  std={result['std_reward']:.2f}"
                    f"  ep_len={result['mean_ep_length']:.1f}"
                    f"  wait={result['mean_waiting_time']:.2f}"
                )
                per_seed_records.append(result)

            if per_seed_records:
                agg = _aggregate(per_seed_records)
                agg_path = Path(configs.variant_result_dir(variant)) / split / "summary.json"
                _write_json(agg_path, {
                    "data_dir": str(split_dir),
                    "split_name": split,
                    "variant": variant,
                    "seeds": [r["seed"] for r in per_seed_records],
                    "aggregate": agg,
                    "per_seed": per_seed_records,
                })
                overall[variant][split] = agg

    summary_path = Path(configs.ROOT) / "results" / "summary.json"
    _write_json(summary_path, {
        "variants": variants,
        "splits": splits,
        "seeds": seeds,
        "aggregate": overall,
    })
    print(f"\nSaved overall summary -> {summary_path}")


if __name__ == "__main__":
    main()
