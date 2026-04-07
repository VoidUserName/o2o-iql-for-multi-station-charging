from __future__ import annotations

from dataclasses import dataclass

from simulator_core.models import ChargingRequest


@dataclass(frozen=True)
class DecisionVehicle:
    vehicle_id: int
    station_id: int
    arrival_time: float
    total_charge_demand: float
    downstream_stations: tuple[int, ...] = ()


@dataclass(frozen=True)
class ChargingDecision:
    first_charge_duration: float
    second_station_id: int | None = None
    second_charge_duration: float = 0.0


@dataclass(frozen=True)
class SecondLegPlan:
    vehicle_id: int
    target_station_id: int
    charge_duration: float


class SplitPlanner:
    def translate(
        self,
        current_ev: DecisionVehicle,
        decision: ChargingDecision,
    ) -> tuple[ChargingRequest, SecondLegPlan | None]:
        total_demand = float(current_ev.total_charge_demand)
        first_duration = float(decision.first_charge_duration)
        second_duration = float(decision.second_charge_duration)
        second_station_id = decision.second_station_id

        if first_duration <= 0.0:
            raise ValueError("first_charge_duration must be > 0.")
        if second_station_id is None:
            if abs(first_duration - total_demand) > 1e-6:
                raise ValueError("No-split decision must allocate all charge to the first leg.")
            if second_duration != 0.0:
                raise ValueError("No-split decision cannot have second_charge_duration.")
            return (
                ChargingRequest(
                    vehicle_id=int(current_ev.vehicle_id),
                    station_id=int(current_ev.station_id),
                    charge_duration=float(first_duration),
                    arrival_time=float(current_ev.arrival_time),
                ),
                None,
            )

        if int(second_station_id) not in {int(station_id) for station_id in current_ev.downstream_stations}:
            raise ValueError("second_station_id must be one of the downstream stations.")
        if second_duration <= 0.0:
            raise ValueError("Split decision must allocate positive second_charge_duration.")
        if abs((first_duration + second_duration) - total_demand) > 1e-6:
            raise ValueError("Split decision durations must sum to total_charge_demand.")

        first_request = ChargingRequest(
            vehicle_id=int(current_ev.vehicle_id),
            station_id=int(current_ev.station_id),
            charge_duration=float(first_duration),
            arrival_time=float(current_ev.arrival_time),
        )
        second_leg_plan = SecondLegPlan(
            vehicle_id=int(current_ev.vehicle_id),
            target_station_id=int(second_station_id),
            charge_duration=float(second_duration),
        )
        return first_request, second_leg_plan
