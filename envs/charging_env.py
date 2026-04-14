from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.maskable_actions import (
    decode_maskable_action,
    frac_from_bin,
    iter_valid_maskable_actions,
    no_split_action_int,
)
from simulator.commitment import Commitment
from simulator.models import ChargingAssignment, ChargingRequest, StationSpec
from simulator.orchestrator import SplitChargingOrchestrator
from simulator.planner import ChargingDecision, DecisionVehicle
from simulator.simulator import SimulatorCore


@dataclass(frozen=True)
class Vehicle:
    vid: int
    arrival_time: float
    route: list[int]
    duration: float


@dataclass(order=True, frozen=True)
class SecondLegArrivalEvent:
    time: float
    priority: int
    vehicle_id: int


@dataclass(frozen=True)
class EnvState:
    submitted_requests: tuple[ChargingRequest, ...]
    active_commitments: tuple[Commitment, ...]
    second_leg_events: tuple[SecondLegArrivalEvent, ...]
    arrival_source_state: Any
    pending_vehicle: Vehicle | None
    clock: float
    total_wait: float
    vehicle_total_wait: tuple[tuple[int, float], ...]
    rng_state: dict[str, Any] | None


class ArrivalSource(Protocol):
    def reset(self) -> None: ...

    def peek(self) -> Vehicle | None: ...

    def pop(self) -> Vehicle | None: ...

    def get_state(self) -> Any: ...

    def set_state(self, state: Any) -> None: ...


class ListArrivalSource:
    def __init__(self, vehicles: list[Vehicle]) -> None:
        self._vehicles = tuple(
            sorted(
                (_normalize_vehicle(vehicle) for vehicle in vehicles),
                key=lambda item: (float(item.arrival_time), int(item.vid)),
            )
        )
        self._index = 0

    def reset(self) -> None:
        self._index = 0

    def peek(self) -> Vehicle | None:
        if self._index >= len(self._vehicles):
            return None
        return self._vehicles[self._index]

    def pop(self) -> Vehicle | None:
        vehicle = self.peek()
        if vehicle is None:
            return None
        self._index += 1
        return vehicle

    def get_state(self) -> int:
        return int(self._index)

    def set_state(self, state: Any) -> None:
        index = int(state)
        if index < 0 or index > len(self._vehicles):
            raise ValueError("arrival source index is out of range.")
        self._index = index


def _normalize_vehicle(vehicle: Vehicle) -> Vehicle:
    normalized = Vehicle(
        vid=int(vehicle.vid),
        arrival_time=float(vehicle.arrival_time),
        route=[int(station_id) for station_id in vehicle.route],
        duration=float(vehicle.duration),
    )
    if not normalized.route:
        raise ValueError("vehicle.route must contain at least one station.")
    if normalized.duration <= 0.0:
        raise ValueError("vehicle.duration must be positive.")
    return normalized


class MultiStationChargingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        vehicles: list[Vehicle] | None = None,
        station_capacities: list[int] | None = None,
        travel_time_fn: Callable[[int, int, Vehicle], float] | None = None,
        min_first_charge: float = 10.0,
        min_second_charge: float = 10.0,
        reward_scale: float = 1.0,
        n_bins: int = 21,
        arrival_source: ArrivalSource | None = None,
        second_leg_arrival_noise_scale: float = 0.25,
        invalid_action_penalty: float = 0.0,
    ) -> None:
        super().__init__()

        if station_capacities is None or not station_capacities:
            raise ValueError("station_capacities must contain at least one station.")
        if vehicles is not None and arrival_source is not None:
            raise ValueError("Provide either vehicles or arrival_source, not both.")
        if vehicles is None and arrival_source is None:
            raise ValueError("Provide vehicles or arrival_source.")
        if int(n_bins) < 2:
            raise ValueError("n_bins must be at least 2.")

        self.station_capacities = [int(capacity) for capacity in station_capacities]
        if any(capacity <= 0 for capacity in self.station_capacities):
            raise ValueError("station capacities must all be positive.")
        self.num_stations = len(self.station_capacities)
        self._station_specs = [
            StationSpec(station_id=station_id, charge_capacity=capacity)
            for station_id, capacity in enumerate(self.station_capacities)
        ]
        self.min_first_charge = float(min_first_charge)
        self.min_second_charge = float(min_second_charge)
        self.reward_scale = float(reward_scale)
        self.n_bins = int(n_bins)
        self.second_leg_arrival_noise_scale = float(second_leg_arrival_noise_scale)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.travel_time_fn = travel_time_fn or (lambda _a, _b, _vehicle: 0.0)

        self._base_vehicles = None if vehicles is None else [_normalize_vehicle(vehicle) for vehicle in vehicles]
        self._arrival_source = arrival_source or ListArrivalSource(self._base_vehicles or [])

        self.action_space = spaces.Discrete((self.num_stations + 1) * self.n_bins)
        self.observation_space = self._build_observation_space()

        self._travel_time_vehicle_context: Vehicle | None = None
        self._sim: SimulatorCore
        self._orchestrator: SplitChargingOrchestrator
        self._submitted_requests: list[ChargingRequest]
        self._second_leg_events: list[SecondLegArrivalEvent]
        self.pending_vehicle: Vehicle | None
        self.clock: float
        self.total_wait: float
        self._vehicle_total_wait: dict[int, float]
        self._reset_runtime()

    def _build_observation_space(self) -> spaces.Dict:
        max_station_id = self.num_stations - 1

        def scalar_float(low: float = 0.0) -> spaces.Box:
            return spaces.Box(
                low=np.array(low, dtype=np.float32),
                high=np.array(np.inf, dtype=np.float32),
                shape=(),
                dtype=np.float32,
            )

        def scalar_int(low: int = 0, high: int = np.iinfo(np.int32).max) -> spaces.Box:
            return spaces.Box(
                low=np.array(low, dtype=np.int32),
                high=np.array(high, dtype=np.int32),
                shape=(),
                dtype=np.int32,
            )

        station_spaces: dict[int, spaces.Space[Any]] = {}
        for station_id, capacity in enumerate(self.station_capacities):
            station_spaces[station_id] = spaces.Dict(
                {
                    "charger_status": spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(capacity,),
                        dtype=np.float32,
                    ),
                    "queue_waiting_time": spaces.Sequence(scalar_float(low=0.0)),
                    "queue_demand": spaces.Sequence(scalar_float(low=0.0)),
                }
            )

        metric_space = spaces.Dict(
            {
                "ev_served": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max,
                    shape=(self.num_stations,),
                    dtype=np.int32,
                ),
                "ev_queueing": spaces.Box(
                    low=0,
                    high=np.iinfo(np.int32).max,
                    shape=(self.num_stations,),
                    dtype=np.int32,
                ),
                "queue_time": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.num_stations,),
                    dtype=np.float32,
                ),
            }
        )

        return spaces.Dict(
            {
                "sim_state": spaces.Dict(
                    {
                        "clock": scalar_float(low=0.0),
                        "stations": spaces.Dict(station_spaces),
                        "metrics": metric_space,
                    }
                ),
                "commitment_features": spaces.Dict(
                    {
                        "commitment_count": spaces.Box(
                            low=0,
                            high=np.iinfo(np.int32).max,
                            shape=(self.num_stations,),
                            dtype=np.int32,
                        ),
                        "commitment_charge_demand": spaces.Box(
                            low=0.0,
                            high=np.inf,
                            shape=(self.num_stations,),
                            dtype=np.float32,
                        ),
                        "earliest_expected_arrival_eta": spaces.Box(
                            low=-1.0,
                            high=np.inf,
                            shape=(self.num_stations,),
                            dtype=np.float32,
                        ),
                    }
                ),
                "current_ev": spaces.Dict(
                    {
                        "station_id": scalar_int(low=0, high=max_station_id),
                        "total_charge_demand": scalar_float(low=0.0),
                        "downstream_stations": spaces.Sequence(spaces.Discrete(self.num_stations)),
                    }
                ),
                "future_demand": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(self.num_stations,),
                    dtype=np.float32,
                ),
            }
        )

    def _reset_runtime(self) -> None:
        self._sim = SimulatorCore(self._station_specs)
        self._orchestrator = SplitChargingOrchestrator(
            simulator=self._sim,
            travel_time_estimator=self._estimate_travel_time,
        )
        self._arrival_source.reset()
        self._submitted_requests = []
        self._second_leg_events = []
        self.pending_vehicle = None
        self.clock = 0.0
        self.total_wait = 0.0
        self._vehicle_total_wait = {}
        self._travel_time_vehicle_context = None

    def _estimate_travel_time(self, from_station: int, to_station: int) -> float:
        vehicle = self._travel_time_vehicle_context
        if vehicle is None:
            vehicle = Vehicle(
                vid=-1,
                arrival_time=self.clock,
                route=[int(from_station), int(to_station)],
                duration=0.0,
            )
        estimate = float(self.travel_time_fn(int(from_station), int(to_station), vehicle))
        return max(0.0, estimate)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        del options
        super().reset(seed=seed)
        self._reset_runtime()
        self._advance_until_next_decision()
        return self._current_observation(), {"total_wait": float(self.total_wait)}

    def step(self, action: int):
        if self.pending_vehicle is None:
            raise AssertionError("No pending decision is available.")

        action_int = self._coerce_action(action)
        vehicle = self.pending_vehicle
        old_queue_total = self._queue_time_total(self.clock)
        current_ev = self._to_decision_vehicle(vehicle)
        decision, invalid_action = self._decision_from_action(vehicle, action_int)

        self._travel_time_vehicle_context = vehicle
        try:
            result = self._orchestrator.apply_decision(current_ev=current_ev, decision=decision)
        finally:
            self._travel_time_vehicle_context = None

        first_request = result["first_request"]
        first_assignment = result["first_assignment"]
        commitment = result["commitment"]

        self._submitted_requests.append(first_request)
        self.total_wait += float(first_assignment.wait_time)
        self._record_vehicle_wait(vehicle.vid, first_assignment.wait_time)

        if commitment is not None:
            actual_arrival_time = self._resolve_second_leg_arrival_time(
                vehicle=vehicle,
                first_assignment=first_assignment,
                target_station_id=int(commitment.target_station_id),
            )
            heapq.heappush(
                self._second_leg_events,
                SecondLegArrivalEvent(
                    time=float(actual_arrival_time),
                    priority=0,
                    vehicle_id=int(vehicle.vid),
                ),
            )

        self.pending_vehicle = None
        self._advance_until_next_decision()

        terminated = self._is_terminated()
        truncated = False
        new_queue_total = self._queue_time_total(self.clock)
        queue_time_delta = float(new_queue_total - old_queue_total)
        reward = (-self.reward_scale * queue_time_delta) - (
            self.invalid_action_penalty if invalid_action else 0.0
        )

        info = {
            "clock": float(self.clock),
            "queue_time_total": float(new_queue_total),
            "queue_time_delta": float(queue_time_delta),
            "total_wait": float(self.total_wait),
            "vehicle_total_wait": float(self._vehicle_total_wait.get(int(vehicle.vid), 0.0)),
            "invalid_action": bool(invalid_action),
            "vid": int(vehicle.vid),
        }
        if terminated:
            info.update(self.compute_episode_metrics())

        return self._current_observation(), reward, terminated, truncated, info

    def _current_observation(self) -> dict[str, Any]:
        if self.pending_vehicle is None:
            return self._terminal_observation()

        current_ev = self._to_decision_vehicle(self.pending_vehicle)
        self._travel_time_vehicle_context = self.pending_vehicle
        try:
            observation = self._orchestrator.build_observation(
                current_ev=current_ev,
                now=float(self.clock),
            )
            return self._filter_observation(observation)
        finally:
            self._travel_time_vehicle_context = None

    def _terminal_observation(self) -> dict[str, Any]:
        self._travel_time_vehicle_context = None
        observation = self._orchestrator.build_observation(
            current_ev=DecisionVehicle(
                vehicle_id=-1,
                station_id=0,
                arrival_time=float(self.clock),
                total_charge_demand=0.0,
                downstream_stations=(),
            ),
            now=float(self.clock),
        )
        return self._filter_observation(observation)

    def _filter_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        sim_state = observation["sim_state"]
        filtered_stations: dict[int, dict[str, Any]] = {}
        for station_id, station_payload in sorted(sim_state["stations"].items()):
            filtered_stations[int(station_id)] = {
                "charger_status": [
                    float(value) for value in station_payload["charger_status"]
                ],
                "queue_waiting_time": tuple(
                    float(value) for value in station_payload["queue_waiting_time"]
                ),
                "queue_demand": tuple(
                    float(value) for value in station_payload["queue_demand"]
                ),
            }

        current_ev = observation["current_ev"]
        return {
            "sim_state": {
                "clock": float(sim_state["clock"]),
                "stations": filtered_stations,
                "metrics": {
                    "ev_served": [
                        int(value) for value in sim_state["metrics"]["ev_served"]
                    ],
                    "ev_queueing": [
                        int(value) for value in sim_state["metrics"]["ev_queueing"]
                    ],
                    "queue_time": [
                        float(value) for value in sim_state["metrics"]["queue_time"]
                    ],
                },
            },
            "commitment_features": {
                "commitment_count": [
                    int(value) for value in observation["commitment_features"]["commitment_count"]
                ],
                "commitment_charge_demand": [
                    float(value)
                    for value in observation["commitment_features"]["commitment_charge_demand"]
                ],
                "earliest_expected_arrival_eta": [
                    float(value)
                    for value in observation["commitment_features"]["earliest_expected_arrival_eta"]
                ],
            },
            "current_ev": {
                "station_id": int(current_ev["station_id"]),
                "total_charge_demand": float(current_ev["total_charge_demand"]),
                "downstream_stations": tuple(
                    int(station_id) for station_id in current_ev["downstream_stations"]
                ),
            },
            "future_demand": [
                float(value) for value in observation["future_demand"]
            ],
        }

    def _coerce_action(self, action: Any) -> int:
        try:
            return int(action)
        except (TypeError, ValueError) as exc:
            raise TypeError("MultiStationChargingEnv only accepts discrete integer actions.") from exc

    def _to_decision_vehicle(self, vehicle: Vehicle) -> DecisionVehicle:
        return DecisionVehicle(
            vehicle_id=int(vehicle.vid),
            station_id=int(vehicle.route[0]),
            arrival_time=float(self.clock),
            total_charge_demand=float(vehicle.duration),
            downstream_stations=tuple(int(station_id) for station_id in vehicle.route[1:]),
        )

    def _decision_from_action(self, vehicle: Vehicle, action_int: int) -> tuple[ChargingDecision, bool]:
        second_choice, frac_bin = decode_maskable_action(
            action_int=action_int,
            n_bins=self.n_bins,
            num_stations=self.num_stations,
        )
        total_demand = float(vehicle.duration)
        canonical_no_split = no_split_action_int(
            n_bins=self.n_bins,
            num_stations=self.num_stations,
        )
        if action_int == canonical_no_split:
            return (
                ChargingDecision(
                    first_charge_duration=total_demand,
                    second_station_id=None,
                    second_charge_duration=0.0,
                ),
                False,
            )

        if second_choice == self.num_stations:
            return (
                ChargingDecision(
                    first_charge_duration=total_demand,
                    second_station_id=None,
                    second_charge_duration=0.0,
                ),
                True,
            )

        first_duration = frac_from_bin(frac_bin, self.n_bins) * total_demand
        second_duration = total_demand - first_duration
        downstream_stations = {int(station_id) for station_id in vehicle.route[1:]}
        is_valid = (
            second_choice in downstream_stations
            and first_duration + 1e-9 >= self.min_first_charge
            and second_duration + 1e-9 >= self.min_second_charge
            and second_duration > 1e-9
        )
        if is_valid:
            return (
                ChargingDecision(
                    first_charge_duration=float(first_duration),
                    second_station_id=int(second_choice),
                    second_charge_duration=float(second_duration),
                ),
                False,
            )

        return (
            ChargingDecision(
                first_charge_duration=total_demand,
                second_station_id=None,
                second_charge_duration=0.0,
            ),
            True,
        )

    def _resolve_second_leg_arrival_time(
        self,
        vehicle: Vehicle,
        first_assignment: ChargingAssignment,
        target_station_id: int,
    ) -> float:
        estimated_eta = float(first_assignment.end_time) + self._travel_time_with_vehicle(
            from_station=int(vehicle.route[0]),
            to_station=int(target_station_id),
            vehicle=vehicle,
        )
        noise = 0.0
        if self.second_leg_arrival_noise_scale > 0.0:
            noise = float(
                self.np_random.uniform(
                    low=-self.second_leg_arrival_noise_scale,
                    high=self.second_leg_arrival_noise_scale,
                )
            )
        return max(float(first_assignment.end_time), float(estimated_eta + noise))

    def _travel_time_with_vehicle(self, from_station: int, to_station: int, vehicle: Vehicle) -> float:
        self._travel_time_vehicle_context = vehicle
        try:
            return self._estimate_travel_time(from_station, to_station)
        finally:
            self._travel_time_vehicle_context = None

    def _advance_until_next_decision(self) -> None:
        while True:
            next_vehicle = self._arrival_source.peek()
            next_vehicle_time = (
                float(next_vehicle.arrival_time) if next_vehicle is not None else float("inf")
            )
            next_second_leg_time = (
                float(self._second_leg_events[0].time)
                if self._second_leg_events
                else float("inf")
            )

            if np.isinf(next_vehicle_time) and np.isinf(next_second_leg_time):
                self.pending_vehicle = None
                return

            if next_second_leg_time <= next_vehicle_time:
                event = heapq.heappop(self._second_leg_events)
                if event.time < self.clock:
                    raise ValueError("second-leg events must be processed in non-decreasing time order.")
                self.clock = float(event.time)
                self._process_second_leg_event(event)
                continue

            vehicle = self._arrival_source.pop()
            if vehicle is None:
                self.pending_vehicle = None
                return
            if float(vehicle.arrival_time) < self.clock:
                raise ValueError("arrival source produced a vehicle earlier than the current env clock.")
            self.clock = float(vehicle.arrival_time)
            self.pending_vehicle = vehicle
            return

    def _process_second_leg_event(self, event: SecondLegArrivalEvent) -> None:
        commitment = self._orchestrator.commitment_store.get(event.vehicle_id)
        if commitment is None:
            return

        second_request = ChargingRequest(
            vehicle_id=int(commitment.vehicle_id),
            station_id=int(commitment.target_station_id),
            charge_duration=float(commitment.planned_second_charge_duration),
            arrival_time=float(event.time),
        )
        second_assignment = self._orchestrator.submit_second_leg_arrival(
            vehicle_id=int(event.vehicle_id),
            actual_arrival_time=float(event.time),
        )
        self._submitted_requests.append(second_request)
        self.total_wait += float(second_assignment.wait_time)
        self._record_vehicle_wait(event.vehicle_id, second_assignment.wait_time)

    def _queue_time_total(self, query_time: float) -> float:
        return float(sum(self._sim.get_metrics(query_time=query_time).queue_time))

    def _record_vehicle_wait(self, vehicle_id: int, wait_time: float) -> None:
        key = int(vehicle_id)
        self._vehicle_total_wait[key] = self._vehicle_total_wait.get(key, 0.0) + float(wait_time)

    def _is_terminated(self) -> bool:
        return (
            self.pending_vehicle is None
            and not self._second_leg_events
            and self._arrival_source.peek() is None
        )

    def action_masks(self) -> np.ndarray:
        if self.pending_vehicle is None:
            return np.ones(self.action_space.n, dtype=np.int8)

        mask = np.zeros(self.action_space.n, dtype=np.int8)
        for action_int in iter_valid_maskable_actions(
            route=self.pending_vehicle.route,
            n_bins=self.n_bins,
            total_duration=float(self.pending_vehicle.duration),
            t_first_min=float(self.min_first_charge),
            t_second_min=float(self.min_second_charge),
            num_stations=self.num_stations,
        ):
            mask[int(action_int)] = 1
        return mask

    def compute_episode_metrics(self) -> dict[str, Any]:
        vehicle_wait_times = np.asarray(
            list(self._vehicle_total_wait.values()),
            dtype=np.float32,
        )
        station_loads = np.asarray(
            self._sim.get_metrics(query_time=self.clock).ev_served[: self.num_stations],
            dtype=np.float32,
        )
        load_mean = float(station_loads.mean()) if station_loads.size > 0 else 0.0
        return {
            "vehicle_wait_times": vehicle_wait_times.astype(float).tolist(),
            "mean_waiting_time": (
                float(vehicle_wait_times.mean()) if vehicle_wait_times.size > 0 else 0.0
            ),
            "p95_waiting_time": (
                float(np.percentile(vehicle_wait_times, 95))
                if vehicle_wait_times.size > 0
                else 0.0
            ),
            "max_waiting_time": (
                float(vehicle_wait_times.max()) if vehicle_wait_times.size > 0 else 0.0
            ),
            "station_loads": station_loads.astype(int).tolist(),
            "load_imbalance": (
                float(station_loads.std() / (load_mean + 1e-8))
                if station_loads.size > 0
                else 0.0
            ),
        }

    def get_state(self) -> EnvState:
        rng_state = None if self.np_random is None else dict(self.np_random.bit_generator.state)
        return EnvState(
            submitted_requests=tuple(self._submitted_requests),
            active_commitments=tuple(self._orchestrator.commitment_store._commitments.values()),  # noqa: SLF001
            second_leg_events=tuple(sorted(self._second_leg_events)),
            arrival_source_state=self._arrival_source.get_state(),
            pending_vehicle=self.pending_vehicle,
            clock=float(self.clock),
            total_wait=float(self.total_wait),
            vehicle_total_wait=tuple(sorted(self._vehicle_total_wait.items())),
            rng_state=rng_state,
        )

    def set_state(self, state: EnvState) -> None:
        self._reset_runtime()
        self._arrival_source.set_state(state.arrival_source_state)
        self._submitted_requests = list(state.submitted_requests)
        for request in self._submitted_requests:
            self._sim.submit_arrival(request)
        for commitment in state.active_commitments:
            self._orchestrator.commitment_store.add(commitment)
        self._second_leg_events = list(state.second_leg_events)
        heapq.heapify(self._second_leg_events)
        self.pending_vehicle = state.pending_vehicle
        self.clock = float(state.clock)
        self.total_wait = float(state.total_wait)
        self._vehicle_total_wait = {
            int(vehicle_id): float(wait_time)
            for vehicle_id, wait_time in state.vehicle_total_wait
        }
        if state.rng_state is not None and self.np_random is not None:
            self.np_random.bit_generator.state = state.rng_state


class EpisodeBankChargingEnv(MultiStationChargingEnv):
    def __init__(
        self,
        episode_bank: list[list[Vehicle]],
        station_capacities: list[int],
        travel_time_fn: Callable[[int, int, Vehicle], float] | None = None,
        min_first_charge: float = 10.0,
        min_second_charge: float = 10.0,
        reward_scale: float = 1.0,
        n_bins: int = 21,
        second_leg_arrival_noise_scale: float = 0.25,
        invalid_action_penalty: float = 0.0,
    ) -> None:
        if not episode_bank:
            raise ValueError("episode_bank must not be empty.")
        self.episode_bank = [
            [_normalize_vehicle(vehicle) for vehicle in episode]
            for episode in episode_bank
        ]
        self.current_episode_index = 0
        super().__init__(
            vehicles=self.episode_bank[0],
            station_capacities=station_capacities,
            travel_time_fn=travel_time_fn,
            min_first_charge=min_first_charge,
            min_second_charge=min_second_charge,
            reward_scale=reward_scale,
            n_bins=n_bins,
            second_leg_arrival_noise_scale=second_leg_arrival_noise_scale,
            invalid_action_penalty=invalid_action_penalty,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            gym.Env.reset(self, seed=seed)
        self.current_episode_index = int(self.np_random.integers(len(self.episode_bank)))
        self._arrival_source = ListArrivalSource(self.episode_bank[self.current_episode_index])
        return super().reset(seed=None, options=options)


def make_travel_time_matrix_default(num_stations: int = 7) -> np.ndarray:
    return np.zeros((int(num_stations), int(num_stations)), dtype=float)


def travel_time_fn_from_matrix(mat: np.ndarray) -> Callable[[int, int, Vehicle], float]:
    def _fn(from_station: int, to_station: int, _vehicle: Vehicle) -> float:
        return float(mat[int(from_station), int(to_station)])

    return _fn


def make_env(
    vehicles: list[Vehicle],
    capacities: list[int],
    mat: np.ndarray,
    n_bins: int = 21,
    seed: int = 0,
    min_first_charge: float = 10.0,
    min_second_charge: float = 10.0,
    reward_scale: float = 1.0,
    episode_bank: list[list[Vehicle]] | None = None,
    second_leg_arrival_noise_scale: float = 0.25,
):
    def _init():
        try:
            from stable_baselines3.common.monitor import Monitor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "stable-baselines3 is only required when calling make_env() for RL training."
            ) from exc

        if episode_bank is None:
            env: gym.Env = MultiStationChargingEnv(
                vehicles=vehicles,
                station_capacities=capacities,
                travel_time_fn=travel_time_fn_from_matrix(mat),
                min_first_charge=min_first_charge,
                min_second_charge=min_second_charge,
                reward_scale=reward_scale,
                n_bins=n_bins,
                second_leg_arrival_noise_scale=second_leg_arrival_noise_scale,
            )
        else:
            env = EpisodeBankChargingEnv(
                episode_bank=episode_bank,
                station_capacities=capacities,
                travel_time_fn=travel_time_fn_from_matrix(mat),
                min_first_charge=min_first_charge,
                min_second_charge=min_second_charge,
                reward_scale=reward_scale,
                n_bins=n_bins,
                second_leg_arrival_noise_scale=second_leg_arrival_noise_scale,
            )

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init
