from __future__ import annotations

from dataclasses import asdict

from simulator_core.metrics import build_empty_metrics
from simulator_core.models import (
    ChargingAssignment,
    ChargingRequest,
    StationSpec,
    SystemMetrics,
    VehicleRecord,
    VehicleState,
    VehicleStatus,
)
from simulator_core.station import StationRuntime


class SimulatorCore:
    def __init__(self, station_specs: list[StationSpec]) -> None:
        if not station_specs:
            raise ValueError("station_specs must not be empty.")
        self._station_specs = {int(spec.station_id): spec for spec in station_specs}
        self._stations = {
            int(spec.station_id): StationRuntime(spec)
            for spec in station_specs
        }
        self._metric_size = 1 + max(int(spec.station_id) for spec in station_specs)
        self._metrics = build_empty_metrics(num_stations=self._metric_size)
        self._clock = 0.0
        self._latest_record_by_vehicle: dict[int, VehicleRecord] = {}

    @property
    def clock(self) -> float:
        return float(self._clock)

    @property
    def station_ids(self) -> list[int]:
        return sorted(int(station_id) for station_id in self._station_specs)

    def submit_arrival(self, request: ChargingRequest) -> ChargingAssignment:
        self._validate_request(request)

        station = self._stations[int(request.station_id)]
        charger_id, start_time, end_time, wait_time = station.reserve(
            arrival_time=float(request.arrival_time),
            charge_duration=float(request.charge_duration),
        )
        status_at_arrival = (
            VehicleStatus.QUEUEING if float(wait_time) > 0.0 else VehicleStatus.CHARGING
        )
        assignment = ChargingAssignment(
            vehicle_id=int(request.vehicle_id),
            station_id=int(request.station_id),
            charger_id=int(charger_id),
            arrival_time=float(request.arrival_time),
            start_time=float(start_time),
            end_time=float(end_time),
            wait_time=float(wait_time),
            status_at_arrival=status_at_arrival,
        )
        self._latest_record_by_vehicle[int(request.vehicle_id)] = VehicleRecord(
            assignment=assignment
        )
        self._metrics.ev_served[int(request.station_id)] += 1
        self._metrics.queue_time[int(request.station_id)] += float(wait_time)
        self._clock = max(float(self._clock), float(request.arrival_time))
        return assignment

    def get_state(
        self,
        query_time: float | None = None,
        vehicle_info: bool = False,
    ) -> dict:
        effective_time = float(self._clock if query_time is None else query_time)
        if effective_time < float(self._clock):
            raise ValueError("query_time must be >= the latest processed arrival_time.")

        queue_waits_by_station: dict[int, list[int]] = {
            int(station_id): []
            for station_id in self._stations
        }
        current_queue_counts = [0 for _ in range(self._metric_size)]
        vehicle_states: dict[int, VehicleState] | None = {} if bool(vehicle_info) else None

        for vehicle_id, record in self._latest_record_by_vehicle.items():
            vehicle_state = record.state_at(effective_time)
            if vehicle_state is None:
                continue
            if vehicle_states is not None:
                vehicle_states[int(vehicle_id)] = vehicle_state
            if vehicle_state.status is VehicleStatus.QUEUEING:
                station_id = int(vehicle_state.station_id)
                queue_wait = int(
                    max(0.0, float(effective_time) - float(vehicle_state.arrival_time))
                )
                queue_waits_by_station[station_id].append(queue_wait)
                current_queue_counts[station_id] += 1

        metrics = self._metrics.copy()
        metrics.ev_queueing = current_queue_counts
        stations = {
            int(station_id): station.to_state(
                query_time=effective_time,
                queue_waits=sorted(queue_waits_by_station[int(station_id)]),
            )
            for station_id, station in self._stations.items()
        }

        state = {
            "clock": float(effective_time),
            "stations": {
                int(station_id): asdict(station_state)
                for station_id, station_state in stations.items()
            },
            "metrics": self._serialize_metrics(metrics),
        }
        if vehicle_states is not None:
            state["vehicles"] = {
                int(vehicle_id): self._serialize_vehicle_state(vehicle_state)
                for vehicle_id, vehicle_state in vehicle_states.items()
            }
        return state

    def get_metrics(self, query_time: float | None = None) -> SystemMetrics:
        effective_time = float(self._clock if query_time is None else query_time)
        if effective_time < float(self._clock):
            raise ValueError("query_time must be >= the latest processed arrival_time.")

        current_queue_counts = [0 for _ in range(self._metric_size)]
        for record in self._latest_record_by_vehicle.values():
            vehicle_state = record.state_at(effective_time)
            if vehicle_state is None:
                continue
            if vehicle_state.status is VehicleStatus.QUEUEING:
                current_queue_counts[int(vehicle_state.station_id)] += 1

        metrics = self._metrics.copy()
        metrics.ev_queueing = current_queue_counts
        return metrics

    def _validate_request(self, request: ChargingRequest) -> None:
        if int(request.station_id) not in self._stations:
            raise ValueError("station_id is not defined in this simulator.")
        if float(request.charge_duration) <= 0.0:
            raise ValueError("charge_duration must be > 0.")
        if float(request.arrival_time) < float(self._clock):
            raise ValueError(
                "arrival_time must be non-decreasing because the simulator does not store future arrivals."
            )

        existing = self._latest_record_by_vehicle.get(int(request.vehicle_id))
        if existing is not None and float(request.arrival_time) < float(existing.assignment.end_time):
            raise ValueError("vehicle_id already has an unfinished charging reservation.")

    def _serialize_vehicle_state(self, vehicle_state: VehicleState) -> dict:
        payload = asdict(vehicle_state)
        payload["status"] = str(vehicle_state.status.value)
        return payload

    def _serialize_metrics(self, metrics: SystemMetrics) -> dict:
        return {
            "ev_served": list(metrics.ev_served),
            "ev_queueing": list(metrics.ev_queueing),
            "queue_time": list(metrics.queue_time),
        }
