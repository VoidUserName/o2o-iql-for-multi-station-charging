from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "data" / "train_dataset"
DEFAULT_SCENARIOS = ("normal", "bias", "extreme")


def _select_evenly_spaced_rows(rows: list[dict[str, str]], keep_count: int) -> list[dict[str, str]]:
    """Keep exactly ``keep_count`` rows while preserving temporal spread."""
    total = len(rows)
    if keep_count <= 0 or total <= 0:
        return []
    if keep_count >= total:
        return list(rows)

    kept: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        prev_bucket = (idx * keep_count) // total
        next_bucket = ((idx + 1) * keep_count) // total
        if next_bucket > prev_bucket:
            kept.append(dict(row))
    return kept


def _rewrite_episode(csv_path: Path, keep_ratio: float) -> tuple[int, int]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"{csv_path} is missing a CSV header.")
    if "Vehicle_ID" not in fieldnames:
        raise ValueError(f"{csv_path} does not contain a Vehicle_ID column.")

    original_count = len(rows)
    keep_count = max(1, round(original_count * keep_ratio)) if original_count > 0 else 0
    kept_rows = _select_evenly_spaced_rows(rows, keep_count)

    for new_vehicle_id, row in enumerate(kept_rows):
        row["Vehicle_ID"] = str(new_vehicle_id)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    return original_count, len(kept_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reduce the number of vehicles in selected train_dataset scenarios.",
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root directory containing scenario subdirectories of CSV episodes.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        help="Scenario subdirectories to rewrite.",
    )
    parser.add_argument(
        "--keep_ratio",
        type=float,
        default=2.0 / 3.0,
        help="Fraction of vehicles to keep in each episode.",
    )
    args = parser.parse_args()

    if not 0.0 < float(args.keep_ratio) <= 1.0:
        raise ValueError("--keep_ratio must be in the range (0, 1].")

    dataset_root = Path(args.dataset_root).resolve()
    grand_original = 0
    grand_kept = 0

    for scenario in args.scenarios:
        scenario_dir = dataset_root / scenario
        if not scenario_dir.is_dir():
            raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

        episode_paths = sorted(scenario_dir.glob("*.csv"))
        if not episode_paths:
            raise FileNotFoundError(f"No CSV episodes found in: {scenario_dir}")

        scenario_original = 0
        scenario_kept = 0
        for csv_path in episode_paths:
            original_count, kept_count = _rewrite_episode(
                csv_path=csv_path,
                keep_ratio=float(args.keep_ratio),
            )
            scenario_original += original_count
            scenario_kept += kept_count

        grand_original += scenario_original
        grand_kept += scenario_kept
        print(
            f"{scenario}: episodes={len(episode_paths)}, "
            f"vehicles {scenario_original} -> {scenario_kept}"
        )

    print(f"total vehicles {grand_original} -> {grand_kept}")


if __name__ == "__main__":
    main()
