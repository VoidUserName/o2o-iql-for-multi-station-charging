"""Orchestrate the full ablation sweep.

Runs (for each variant in configs.ABLATIONS, for each seed in configs.SEEDS) a
fresh subprocess calling `run_variant.py`. Subprocess isolation keeps RNG state,
CUDA memory, and monkey-patches from leaking across runs.

Usage:
    # Step 1: one-off shared offline pretraining (produces SHARED_CKPT).
    python -m exps.ablations.run_ablations --pretrain

    # Step 2: full sweep (6 variants * 3 seeds = 18 runs).
    python -m exps.ablations.run_ablations

    # Optional filters:
    python -m exps.ablations.run_ablations --variants full_o2o_iql no_ucb
    python -m exps.ablations.run_ablations --seeds 42
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from exps.ablations import configs


def _run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    start = time.time()
    proc = subprocess.run(cmd)
    elapsed = time.time() - start
    print(f"  -> exit={proc.returncode}  elapsed={elapsed / 60:.1f} min", flush=True)
    return int(proc.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full O2O IQL ablation sweep.")
    parser.add_argument("--pretrain", action="store_true",
                        help="Run offline pretraining once and save SHARED_CKPT, then exit.")
    parser.add_argument("--variants", nargs="*", default=None,
                        choices=list(configs.ABLATIONS))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    python = sys.executable

    if args.pretrain:
        rc = _run([python, "-m", "exps.ablations.run_variant", "--pretrain_only"])
        if rc != 0:
            raise SystemExit(rc)
        return

    if not Path(configs.SHARED_CKPT).exists():
        raise SystemExit(
            f"Shared offline ckpt missing at '{configs.SHARED_CKPT}'. "
            f"Run with --pretrain first."
        )

    variants = args.variants or list(configs.ABLATIONS)
    seeds = args.seeds or list(configs.SEEDS)

    total = len(variants) * len(seeds)
    print(f"=== Running {total} ablation training runs "
          f"({len(variants)} variants x {len(seeds)} seeds) ===")

    failures: list[tuple[str, int, int]] = []
    for variant in variants:
        for seed in seeds:
            rc = _run([
                python, "-m", "exps.ablations.run_variant",
                "--variant", variant,
                "--seed", str(int(seed)),
            ])
            if rc != 0:
                failures.append((variant, int(seed), rc))

    print("\n=== Sweep summary ===")
    print(f"  total runs  : {total}")
    print(f"  successful  : {total - len(failures)}")
    print(f"  failed      : {len(failures)}")
    for variant, seed, rc in failures:
        print(f"    - {variant} seed={seed} exit={rc}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
