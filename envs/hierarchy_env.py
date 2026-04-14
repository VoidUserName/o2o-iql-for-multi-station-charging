"""Hierarchical (two-phase) action wrapper for multi-station EV charging.

Decomposes the flat ``Discrete((S+1)*B)`` action space into two phases:

Phase 0 — **High-level** (routing):
    Actions ``[0, S)``  → split charging, send second leg to station *s*
    Action  ``S``       → no-split (charge fully at current station)

Phase 1 — **Low-level** (allocation, only after split):
    Actions ``[S+1, S+B]`` → fraction-bin *b* = action − S − 1
        determines how much charge goes to the first vs second station.

Observation is flattened **and filtered** — static / redundant fields are
removed to give the agent a leaner input:

Removed (vs ``FlatObsWrapper``):
    ✕ ``station_id`` per station (position-encoded, always == index)
    ✕ ``charge_capacity`` per station (static, never changes)
    ✕ ``available_info`` per station (derivable: charger_status == 0)
    ✕ ``vehicle_id`` of current EV (irrelevant for decisions)
    ✕ ``arrival_time`` of current EV (== clock, redundant)
    ✕ ``travel_time_matrix`` (static 7×7 = 49 floats)

Added:
    ✓ ``phase`` indicator (0 or 1)
    ✓ ``high_choice`` one-hot (S+1 dims; all-zero in phase 0)

Net effect: obs dim ≈ 325 (vs ≈ 498 with FlatObsWrapper).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.maskable_actions import (
    encode_maskable_action,
    no_split_action_int,
)


class HierarchicalChargingEnv(gym.Wrapper):
    """Two-phase action wrapper with filtered flat observations.

    Wraps any ``MultiStationChargingEnv`` or ``EpisodeBankChargingEnv``.

    Args:
        env:            Base charging environment.
        max_queue_len:  Max queue entries per station to encode (pad/truncate).
    """

    def __init__(self, env: gym.Env, max_queue_len: int = 10) -> None:
        super().__init__(env)
        self.S: int = int(env.num_stations)
        self.B: int = int(env.n_bins)
        self.max_queue_len = int(max_queue_len)
        self._capacities: list[int] = list(env.station_capacities)

        # Combined action space: S+1 high-level choices + B low-level bins
        self.action_space = spaces.Discrete(self.S + 1 + self.B)

        # Internal phase state
        self._phase = 0          # 0 = high-level, 1 = low-level
        self._high_choice = -1   # station chosen in phase 0
        self._cached_raw_obs: dict | None = None

        # Filtered flat observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim(),),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # gym.Env interface
    # ------------------------------------------------------------------

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phase = 0
        self._high_choice = -1
        self._cached_raw_obs = obs
        return self._make_flat_obs(obs), info

    def step(self, action: int):
        action = int(action)
        S, B = self.S, self.B

        if self._phase == 0:
            if action == S:
                # No-split → execute immediately
                base_act = no_split_action_int(n_bins=B, num_stations=S)
                obs, reward, term, trunc, info = self.env.step(base_act)
                self._cached_raw_obs = obs
                self._high_choice = -1
                return self._make_flat_obs(obs), reward, term, trunc, info

            # Split chosen → enter phase 1 (no base env step yet)
            self._high_choice = action
            self._phase = 1
            return (
                self._make_flat_obs(self._cached_raw_obs),
                0.0, False, False, {},
            )

        # Phase 1: fraction decision → execute combined action
        frac_bin = action - S - 1
        base_act = encode_maskable_action(
            second_choice=self._high_choice,
            frac_bin=frac_bin,
            n_bins=B,
            num_stations=S,
        )
        obs, reward, term, trunc, info = self.env.step(base_act)
        self._cached_raw_obs = obs
        self._phase = 0
        self._high_choice = -1
        return self._make_flat_obs(obs), reward, term, trunc, info

    def action_masks(self) -> np.ndarray:
        """Hierarchical mask derived from the base env's flat mask."""
        S, B = self.S, self.B
        mask = np.zeros(S + 1 + B, dtype=np.int8)
        base_mask = self.env.action_masks()

        if self._phase == 0:
            # No-split
            mask[S] = base_mask[no_split_action_int(n_bins=B, num_stations=S)]
            # Split to station s — valid if any frac_bin for s is valid
            for s in range(S):
                if base_mask[s * B : (s + 1) * B].any():
                    mask[s] = 1
        else:
            # Phase 1: fraction bins for the chosen station
            h = self._high_choice
            mask[S + 1 :] = base_mask[h * B : (h + 1) * B]

        return mask

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_dim(self) -> int:
        S = self.S
        mq = self.max_queue_len
        station_dim = sum(cap + 2 * mq for cap in self._capacities)
        return (
            1                 # clock
            + station_dim     # charger_status + queue_wt + queue_demand
            + 3 * S           # metrics: ev_served, ev_queueing, queue_time
            + 3 * S           # commitment: count, demand, eta
            + 2 + S           # current_ev: station_id, demand, downstream_mask
            + S               # future_demand
            + 1               # phase
            + (S + 1)         # high_choice one-hot
        )

    def _make_flat_obs(self, raw: dict) -> np.ndarray:
        S = self.S
        mq = self.max_queue_len
        parts: list[np.ndarray] = []

        sim = raw["sim_state"]

        # --- clock ---
        parts.append(np.array([float(sim["clock"])], dtype=np.float32))

        # --- per-station (filtered: charger_status, queue_wt, queue_demand) ---
        for sid, cap in enumerate(self._capacities):
            st = sim["stations"][sid]
            parts.append(np.asarray(st["charger_status"], dtype=np.float32))

            qwt = [float(v) for v in st["queue_waiting_time"]][:mq]
            qd  = [float(v) for v in st["queue_demand"]][:mq]
            qwt += [0.0] * (mq - len(qwt))
            qd  += [0.0] * (mq - len(qd))
            parts.append(np.array(qwt, dtype=np.float32))
            parts.append(np.array(qd,  dtype=np.float32))

        # --- metrics ---
        m = sim["metrics"]
        parts.append(np.asarray(m["ev_served"],   dtype=np.float32)[:S])
        parts.append(np.asarray(m["ev_queueing"], dtype=np.float32)[:S])
        parts.append(np.asarray(m["queue_time"],  dtype=np.float32)[:S])

        # --- commitment features ---
        cf = raw["commitment_features"]
        parts.append(np.asarray(cf["commitment_count"],              dtype=np.float32))
        parts.append(np.asarray(cf["commitment_charge_demand"],      dtype=np.float32))
        parts.append(np.asarray(cf["earliest_expected_arrival_eta"], dtype=np.float32))

        # --- current EV (filtered: no vehicle_id, no arrival_time) ---
        ev = raw["current_ev"]
        parts.append(np.array([
            float(ev["station_id"]),
            float(ev["total_charge_demand"]),
        ], dtype=np.float32))
        ds_mask = np.zeros(S, dtype=np.float32)
        for sid in ev["downstream_stations"]:
            if 0 <= int(sid) < S:
                ds_mask[int(sid)] = 1.0
        parts.append(ds_mask)

        # --- future demand ---
        parts.append(np.asarray(raw["future_demand"], dtype=np.float32)[:S])

        # --- hierarchy context ---
        parts.append(np.array([float(self._phase)], dtype=np.float32))
        hc = np.zeros(S + 1, dtype=np.float32)
        if self._high_choice >= 0:
            hc[self._high_choice] = 1.0
        parts.append(hc)

        return np.concatenate(parts)
