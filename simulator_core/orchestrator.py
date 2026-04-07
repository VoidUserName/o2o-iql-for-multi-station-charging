from __future__ import annotations

from dataclasses import asdict
from typing import Callable

from simulator_core.commitment import Commitment, CommitmentStore
from simulator_core.models import ChargingAssignment, ChargingRequest
from simulator_core.planner import ChargingDecision, DecisionVehicle, SplitPlanner
from simulator_core.simulator import SimulatorCore


class SplitChargingOrchestrator:
    def __init__(
        self,
        simulator: SimulatorCore,
        travel_time_estimator: Callable[[int, int], float] | None = None,
        planner: SplitPlanner | None = None,
    ) -> None:
        self.simulator = simulator
        self.travel_time_estimator = travel_time_estimator or (lambda _a, _b: 0.0)
        self.planner = planner or SplitPlanner()
        self.commitment_store = CommitmentStore(station_ids=self.simulator.station_ids)

    def build_observation(
        self,
        current_ev: DecisionVehicle,
        now: float,
        vehicle_info: bool = False,
    ) -> dict:
        return {
            "sim_state": self.simulator.get_state(query_time=now, vehicle_info=vehicle_info),
            "commitment_features": self.commitment_store.summary(now=now),
            "current_ev": asdict(current_ev),
        }

    def apply_decision(
        self,
        current_ev: DecisionVehicle,
        decision: ChargingDecision,
    ) -> dict:
        first_request, second_leg_plan = self.planner.translate(
            current_ev=current_ev,
            decision=decision,
        )
        first_assignment = self.simulator.submit_arrival(first_request)

        commitment = None
        if second_leg_plan is not None:
            expected_arrival_time = float(first_assignment.end_time) + float(
                self.travel_time_estimator(
                    int(current_ev.station_id),
                    int(second_leg_plan.target_station_id),
                )
            )
            commitment = Commitment(
                vehicle_id=int(second_leg_plan.vehicle_id),
                target_station_id=int(second_leg_plan.target_station_id),
                expected_arrival_time=float(expected_arrival_time),
                planned_second_charge_duration=float(second_leg_plan.charge_duration),
                created_at=float(current_ev.arrival_time),
            )
            self.commitment_store.add(commitment)

        return {
            "first_request": first_request,
            "first_assignment": first_assignment,
            "commitment": commitment,
        }

    def submit_second_leg_arrival(
        self,
        vehicle_id: int,
        actual_arrival_time: float,
    ) -> ChargingAssignment:
        commitment = self.commitment_store.pop(vehicle_id)
        second_request = ChargingRequest(
            vehicle_id=int(commitment.vehicle_id),
            station_id=int(commitment.target_station_id),
            charge_duration=float(commitment.planned_second_charge_duration),
            arrival_time=float(actual_arrival_time),
        )
        return self.simulator.submit_arrival(second_request)
