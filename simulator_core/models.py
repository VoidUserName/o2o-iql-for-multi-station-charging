from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VehicleStatus(Enum):
    QUEUEING = "queueing"
    CHARGING = "charging"
    COMPLETE = "complete"


@dataclass(frozen=True)
class StationSpec:
    station_id: int
    charge_capacity: int
    limits: dict | None = None


@dataclass(frozen=True)
class ChargingRequest:
    vehicle_id: int
    station_id: int
    charge_duration: float
    arrival_time: float


@dataclass(frozen=True)
class ChargingAssignment:
    vehicle_id: int
    station_id: int
    charger_id: int
    arrival_time: float
    start_time: float
    end_time: float
    wait_time: float
    status_at_arrival: VehicleStatus


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: int
    station_id: int
    arrival_time: float
    charge_duration: float
    start_time: float
    end_time: float
    wait_time: float
    status: VehicleStatus


@dataclass(frozen=True)
class StationState:
    station_id: int
    charge_capacity: int
    charger_status: list[float]
    available_info: list[bool]
    queue: list[int]


@dataclass
class SystemMetrics:
    ev_served: list[int]
    ev_queueing: list[int]
    queue_time: list[float]

    def copy(self) -> SystemMetrics:
        return SystemMetrics(
            ev_served=list(self.ev_served),
            ev_queueing=list(self.ev_queueing),
            queue_time=list(self.queue_time),
        )


@dataclass(frozen=True)
class SystemState:
    clock: float
    stations: dict[int, StationState]
    metrics: SystemMetrics
    # vehicle_info: bool = False
    vehicles: dict[int, VehicleState] | None = None


@dataclass
class StationRuntimeSnapshot:
    station_id: int
    release_times: list[float]


@dataclass
class VehicleRecord:
    assignment: ChargingAssignment

    def state_at(self, query_time: float) -> VehicleState | None:
        if float(query_time) < float(self.assignment.arrival_time):
            return None

        if float(query_time) < float(self.assignment.start_time):
            status = VehicleStatus.QUEUEING
        elif float(query_time) < float(self.assignment.end_time):
            status = VehicleStatus.CHARGING
        else:
            status = VehicleStatus.COMPLETE

        return VehicleState(
            vehicle_id=int(self.assignment.vehicle_id),
            station_id=int(self.assignment.station_id),
            arrival_time=float(self.assignment.arrival_time),
            charge_duration=float(self.assignment.end_time - self.assignment.start_time),
            start_time=float(self.assignment.start_time),
            end_time=float(self.assignment.end_time),
            wait_time=float(self.assignment.wait_time),
            status=status,
        )
