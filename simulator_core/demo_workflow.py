from __future__ import annotations

import json
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulator_core import (
    ChargingDecision,
    DecisionVehicle,
    SimulatorCore,
    SplitChargingOrchestrator,
    StationSpec,
)


def _to_payload(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _to_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: _to_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_payload(item) for item in value]
    return value


def run_workflow_demo() -> dict[str, Any]:
    simulator = SimulatorCore(
        station_specs=[
            StationSpec(station_id=0, charge_capacity=1),
            StationSpec(station_id=1, charge_capacity=1),
            StationSpec(station_id=2, charge_capacity=1),
        ]
    )
    orchestrator = SplitChargingOrchestrator(
        simulator=simulator,
        travel_time_estimator=lambda from_station, to_station: {
            (0, 2): 5.0,
        }.get((from_station, to_station), 0.0),
    )

    first_vehicle = DecisionVehicle(
        vehicle_id=101,
        station_id=0,
        arrival_time=0.0,
        total_charge_demand=10.0,
        downstream_stations=(2,),
    )
    first_decision = ChargingDecision(
        first_charge_duration=4.0,
        second_station_id=2,
        second_charge_duration=6.0,
    )

    first_observation = orchestrator.build_observation(
        current_ev=first_vehicle,
        now=0.0,
    )
    first_decision_result = orchestrator.apply_decision(
        current_ev=first_vehicle,
        decision=first_decision,
    )

    second_vehicle = DecisionVehicle(
        vehicle_id=102,
        station_id=0,
        arrival_time=1.0,
        total_charge_demand=4.0,
        downstream_stations=(),
    )
    second_vehicle_observation = orchestrator.build_observation(
        current_ev=second_vehicle,
        now=1.0,
    )
    second_vehicle_decision = ChargingDecision(
        first_charge_duration=4.0,
        second_station_id=None,
        second_charge_duration=0.0,
    )
    second_decision_result = orchestrator.apply_decision(
        current_ev=second_vehicle,
        decision=second_vehicle_decision,
    )

    second_leg_assignment = orchestrator.submit_second_leg_arrival(
        vehicle_id=101,
        actual_arrival_time=10.0,
    )
    final_state = orchestrator.simulator.get_state(query_time=10.0, vehicle_info=True)

    return {
        "inputs": {
            "station_specs": _to_payload(
                [
                    StationSpec(station_id=0, charge_capacity=1),
                    StationSpec(station_id=1, charge_capacity=1),
                    StationSpec(station_id=2, charge_capacity=1),
                ]
            ),
            "first_vehicle": _to_payload(first_vehicle),
            "first_decision": _to_payload(first_decision),
            "second_vehicle": _to_payload(second_vehicle),
            "second_decision": _to_payload(second_vehicle_decision),
        },
        "steps": {
            "first_observation": _to_payload(first_observation),
            "first_decision": _to_payload(first_decision_result),
            "second_vehicle_observation": _to_payload(second_vehicle_observation),
            "second_decision": _to_payload(second_decision_result),
            "second_leg_arrival": {
                "second_request_assignment": _to_payload(second_leg_assignment),
                "remaining_commitments": _to_payload(
                    orchestrator.commitment_store.summary(now=10.0)
                ),
            },
        },
        "final_state": _to_payload(final_state),
    }


def main() -> None:
    print(json.dumps(run_workflow_demo(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
