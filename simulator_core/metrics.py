from __future__ import annotations

from simulator_core.models import SystemMetrics


def build_empty_metrics(num_stations: int) -> SystemMetrics:
    return SystemMetrics(
        ev_served=[0 for _ in range(num_stations)],
        ev_queueing=[0 for _ in range(num_stations)],
        queue_time=[0.0 for _ in range(num_stations)],
    )
 