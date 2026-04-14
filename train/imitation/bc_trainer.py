"""Offline Behavior Cloning pre-trainer for the EV Charging environment.

Pairs each demand episode with its expert solution (data/bc_dataset/solutions/),
decodes per-station charge allocations into discrete maskable actions, rolls out
the env collecting (obs, action, mask) tuples, then pre-trains a MaskablePPO
policy network via negative-log-likelihood (cross-entropy) loss.

Usage:
    python -m train.imitation.bc_trainer [args]
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO

from data.env_data import CAPACITY, MIN_SEG, TRAVEL_MATRIX
from envs.charging_env import EpisodeBankChargingEnv, travel_time_fn_from_matrix
from envs.maskable_actions import (
    encode_maskable_action,
    no_split_action_int,
)
from simulator.orchestrator import load_demand_vehicles_from_csv
from train.finetune.ppo_trainer import FlatObsWrapper, make_charging_env


# ---------------------------------------------------------------------------
# Solution loading & action decoding
# ---------------------------------------------------------------------------

def _load_solution(path: str | Path) -> dict[int, dict]:
    """Return {vehicle_id: row_dict} from a solution CSV."""
    solutions: dict[int, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = int(row["vehicle_id"])
            solutions[vid] = row
    return solutions


def solution_to_action(
    vehicle_id: int,
    route: list[int],
    required_charge_time: float,
    solution: dict[int, dict],
    n_bins: int,
    num_stations: int,
) -> int:
    """Decode a solution row into a discrete maskable action integer.

    The solution records how much charge each station provides.
    - If only the first station has charge → no-split action.
    - If a downstream station also has charge → split action encoding
      the first-station fraction and the second station id.
    """
    no_split = no_split_action_int(n_bins=n_bins, num_stations=num_stations)

    if vehicle_id not in solution:
        return no_split

    row = solution[vehicle_id]
    first_sid = int(route[0])

    try:
        first_charge = float(row[f"station_{first_sid}"])
    except KeyError:
        return no_split

    # Find downstream stations that received charge
    split_candidates = []
    for sid in route[1:]:
        key = f"station_{sid}"
        if key in row:
            charge = float(row[key])
            if charge > 1e-6:
                split_candidates.append((sid, charge))

    if not split_candidates:
        # No split — all charge at first station
        return no_split

    # Take the first downstream station with charge
    second_sid, _ = split_candidates[0]
    total = float(required_charge_time)
    if total < 1e-9:
        return no_split

    frac = first_charge / total
    frac_bin = int(round(frac * (n_bins - 1)))
    # Clip to the valid split range [0, n_bins-2] (n_bins-1 is reserved for no-split)
    frac_bin = max(0, min(n_bins - 2, frac_bin))

    return encode_maskable_action(
        second_choice=int(second_sid),
        frac_bin=int(frac_bin),
        n_bins=int(n_bins),
        num_stations=int(num_stations),
    )


# ---------------------------------------------------------------------------
# Episode-solution pairing
# ---------------------------------------------------------------------------

def _episode_id_from_filename(name: str) -> str:
    """Extract the zero-padded ID from filenames like 'episode_0012.csv'."""
    stem = Path(name).stem          # 'episode_0012'
    return stem.split("_")[-1]     # '0012'


def load_paired_dataset(
    demand_dir: str,
    solution_dir: str,
) -> list[tuple[list, dict[int, dict]]]:
    """Return [(vehicles, solution_dict), ...] for matched demand/solution pairs."""
    demand_files = {_episode_id_from_filename(f): f
                    for f in sorted(Path(demand_dir).glob("episode_*.csv"))}
    solution_files = {_episode_id_from_filename(f): f
                      for f in sorted(Path(solution_dir).glob("solution_*.csv"))}

    common_ids = sorted(set(demand_files) & set(solution_files))
    if not common_ids:
        raise FileNotFoundError(
            f"No matching episode/solution pairs found in '{demand_dir}' and '{solution_dir}'"
        )

    pairs = []
    for eid in common_ids:
        vehicles = load_demand_vehicles_from_csv(str(demand_files[eid]))
        sol = _load_solution(str(solution_files[eid]))
        pairs.append((vehicles, sol))

    print(f"  Loaded {len(pairs)} paired episodes from bc_dataset")
    return pairs


# ---------------------------------------------------------------------------
# Demonstration collection
# ---------------------------------------------------------------------------

def collect_demonstrations(
    paired_data: list[tuple[list, dict[int, dict]]],
    n_bins: int = 21,
    max_queue_len: int = 50,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out the expert solution policy and collect (obs, action, mask) tuples.

    The env is stepped episode by episode. For each pending vehicle, the expert
    action is decoded from the corresponding solution row.

    Returns:
        observations : (N, obs_dim) float32
        actions      : (N,)         int64
        action_masks : (N, act_dim) float32
    """
    capacities = CAPACITY.tolist()
    num_stations = len(capacities)
    tt_fn = travel_time_fn_from_matrix(TRAVEL_MATRIX)
    min_charge = float(MIN_SEG)

    # Build EpisodeBankChargingEnv (we step each episode manually)
    episode_bank = [vehicles for vehicles, _ in paired_data]
    env = EpisodeBankChargingEnv(
        episode_bank=episode_bank,
        station_capacities=capacities,
        travel_time_fn=tt_fn,
        min_first_charge=min_charge,
        min_second_charge=min_charge,
        n_bins=n_bins,
    )
    env = FlatObsWrapper(env, max_queue_len=max_queue_len)

    observations: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []

    for ep_idx, (_, solution) in enumerate(paired_data):
        obs, _ = env.reset(seed=seed + ep_idx)
        # After reset the env has already advanced to the first pending vehicle.
        # We need access to the underlying env's pending_vehicle for solution lookup.
        base_env = env.unwrapped  # FlatObsWrapper wraps EpisodeBankChargingEnv

        done = False
        while not done:
            pending = base_env.pending_vehicle
            if pending is None:
                break

            mask = env.action_masks()

            # Decode expert action from solution
            action = solution_to_action(
                vehicle_id=int(pending.vid),
                route=list(pending.route),
                required_charge_time=float(pending.duration),
                solution=solution,
                n_bins=n_bins,
                num_stations=num_stations,
            )

            # Validate: fall back to no-split if action is masked out
            if mask[action] == 0:
                action = no_split_action_int(n_bins=n_bins, num_stations=num_stations)

            observations.append(obs.copy())
            actions.append(action)
            masks.append(mask.copy())

            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()

    obs_arr  = np.array(observations, dtype=np.float32)
    act_arr  = np.array(actions,      dtype=np.int64)
    mask_arr = np.array(masks,        dtype=np.float32)

    split_frac = float((act_arr != no_split_action_int(n_bins=n_bins, num_stations=num_stations)).mean())
    print(f"  Collected {len(obs_arr)} demo steps  (split-action rate={split_frac:.1%})")
    return obs_arr, act_arr, mask_arr


# ---------------------------------------------------------------------------
# BC pre-training
# ---------------------------------------------------------------------------

def bc_pretrain(
    model: MaskablePPO,
    observations: np.ndarray,
    actions: np.ndarray,
    masks: np.ndarray,
    n_epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> MaskablePPO:
    """Train MaskablePPO policy network with BC (negative log-likelihood) loss.

    Uses the policy's own ``evaluate_actions`` (which honours action masks)
    to compute log π(a_expert | s), then minimises −log π.

    Args:
        model:        Freshly initialised MaskablePPO whose policy will be
                      updated in-place.
        observations: (N, obs_dim) float32 expert observations.
        actions:      (N,)         int64  expert actions.
        masks:        (N, act_dim) float32 action masks at each step.
        n_epochs:     Number of passes over the demo dataset.
        batch_size:   Mini-batch size.
        lr:           Learning rate for the BC Adam optimiser.

    Returns:
        model with pre-trained policy network.
    """
    policy = model.policy
    policy.set_training_mode(True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    obs_t  = torch.FloatTensor(observations).to(model.device)
    act_t  = torch.LongTensor(actions).to(model.device)

    # masks stay as np arrays — evaluate_actions expects np.ndarray
    dataset = TensorDataset(obs_t, act_t, torch.arange(len(obs_t)))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    best_loss = float("inf")
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for obs_b, act_b, idx_b in loader:
            mask_b = masks[idx_b.numpy()]   # (B, act_dim) np array

            _, log_prob, _ = policy.evaluate_actions(obs_b, act_b, action_masks=mask_b)
            loss = -log_prob.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs}  bc_loss={avg_loss:.4f}  best={best_loss:.4f}")

    policy.set_training_mode(False)
    return model


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------

def train_bc(args: argparse.Namespace) -> None:
    print("=== Offline Behavior Cloning Pre-training ===")

    paired_data = load_paired_dataset(
        demand_dir=args.demand_dir,
        solution_dir=args.solution_dir,
    )

    observations, actions, masks = collect_demonstrations(
        paired_data=paired_data,
        n_bins=args.n_bins,
        max_queue_len=args.max_queue_len,
        seed=args.seed,
    )

    # Build a dummy vec-env just so MaskablePPO can infer obs / action dims
    episode_bank = [vehicles for vehicles, _ in paired_data]
    env_fn = make_charging_env(
        episode_bank=episode_bank,
        n_bins=args.n_bins,
        max_queue_len=args.max_queue_len,
        seed=args.seed,
    )
    vec_env = DummyVecEnv([env_fn])

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        verbose=0,
        seed=args.seed,
    )

    print(f"  obs_dim={observations.shape[1]}  act_dim={vec_env.action_space.n}")
    print(f"  n_demos={len(observations)}  n_epochs={args.n_epochs}  lr={args.lr}")

    bc_pretrain(
        model=model,
        observations=observations,
        actions=actions,
        masks=masks,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    os.makedirs(args.save_path, exist_ok=True)
    out_path = os.path.join(args.save_path, "bc_pretrained")
    model.save(out_path)
    print(f"\nBC pre-trained model saved → {out_path}.zip")
    vec_env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline BC pre-training for EV Charging")
    p.add_argument("--demand_dir",    type=str,   default="data/bc_dataset/demand",
                   help="Directory of demand episode CSVs")
    p.add_argument("--solution_dir",  type=str,   default="data/bc_dataset/solutions",
                   help="Directory of expert solution CSVs")
    p.add_argument("--n_bins",        type=int,   default=21)
    p.add_argument("--max_queue_len", type=int,   default=10)
    p.add_argument("--n_epochs",      type=int,   default=50,
                   help="BC training epochs over the demo dataset")
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--save_path",     type=str,   default="train/imitation/checkpoints",
                   help="Directory to save the pre-trained model")
    return p.parse_args()


if __name__ == "__main__":
    train_bc(parse_args())
