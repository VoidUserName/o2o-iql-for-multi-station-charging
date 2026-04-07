from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Commitment:
    vehicle_id: int
    target_station_id: int
    expected_arrival_time: float
    planned_second_charge_duration: float
    created_at: float


class CommitmentStore:
    def __init__(self, station_ids: list[int]) -> None:
        if not station_ids:
            raise ValueError("station_ids must not be empty.")
        self._station_ids = sorted(int(station_id) for station_id in station_ids)
        self._metric_size = 1 + max(self._station_ids)
        self._commitments: dict[int, Commitment] = {}

    def add(self, commitment: Commitment) -> None:
        self._commitments[int(commitment.vehicle_id)] = commitment

    def get(self, vehicle_id: int) -> Commitment | None:
        return self._commitments.get(int(vehicle_id))

    def pop(self, vehicle_id: int) -> Commitment:
        key = int(vehicle_id)
        if key not in self._commitments:
            raise KeyError(f"No active commitment for vehicle_id={key}.")
        return self._commitments.pop(key)

    def summary(self, now: float) -> dict:
        counts = [0 for _ in range(self._metric_size)]
        charge_demand = [0.0 for _ in range(self._metric_size)]
        earliest_eta = [-1.0 for _ in range(self._metric_size)]

        for commitment in self._commitments.values():
            station_id = int(commitment.target_station_id)
            counts[station_id] += 1
            charge_demand[station_id] += float(commitment.planned_second_charge_duration)
            eta = max(0.0, float(commitment.expected_arrival_time) - float(now))
            current_eta = earliest_eta[station_id]
            if current_eta < 0.0 or eta < current_eta:
                earliest_eta[station_id] = float(eta)

        return {
            "commitment_count": counts,
            "commitment_charge_demand": charge_demand,
            "earliest_expected_arrival_eta": earliest_eta,
        }
