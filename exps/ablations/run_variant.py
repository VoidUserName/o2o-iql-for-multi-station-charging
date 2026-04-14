"""Run a single O2O IQL ablation variant.

Usage:
    python -m exps.ablations.run_variant --variant <key> --seed <int>
    python -m exps.ablations.run_variant --pretrain_only

This script never modifies any file under `train/`. It composes an argv for
`train.o2o_iql.trainer.main()` from the variant config, applies an optional
runtime patch to DiscreteIQLAgent, and calls main() in-process.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from exps.ablations import configs


def _apply_patch(patch_name: str | None) -> None:
    if patch_name is None:
        return
    if patch_name == "actor_stochastic":
        from train.iql.agent import DiscreteIQLAgent

        def _patched_act_ucb(self, observation, action_mask, ucb_coef: float = 1.0):
            return self.act(observation, action_mask, deterministic=False)

        DiscreteIQLAgent.act_ucb = _patched_act_ucb  # type: ignore[assignment]
        print(
            f"  [patch] DiscreteIQLAgent.act_ucb -> {DiscreteIQLAgent.act_ucb.__qualname__}"
            f" (actor stochastic sampling)"
        )
        return
    raise ValueError(f"Unknown patch name: {patch_name}")


def _base_argv(save_path: str, log_dir: str, seed: int) -> list[str]:
    return [
        "--train_data_dir", configs.TRAIN_DATA_DIR,
        "--save_path", save_path,
        "--log_dir", log_dir,
        "--seed", str(int(seed)),
        "--online_steps", str(int(configs.ONLINE_STEPS)),
        "--offline_epochs", str(int(configs.OFFLINE_EPOCHS)),
    ]


def _variant_argv(variant: str, seed: int) -> tuple[list[str], str | None]:
    spec = configs.ABLATIONS[variant]
    save_path = configs.variant_save_path(variant, seed)
    log_dir = configs.variant_log_dir(variant, seed)
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    argv = _base_argv(save_path=save_path, log_dir=log_dir, seed=seed)

    if spec.get("use_pretrained"):
        if not Path(configs.SHARED_CKPT).exists():
            raise FileNotFoundError(
                f"Shared offline checkpoint not found at '{configs.SHARED_CKPT}'. "
                f"Run `python -m exps.ablations.run_variant --pretrain_only` first."
            )
        argv += ["--pretrained_checkpoint", configs.SHARED_CKPT]

    for key, value in spec.get("cli", {}).items():
        argv += [f"--{key}", str(value)]

    return argv, spec.get("patch")


def _pretrain_argv() -> list[str]:
    save_path = configs.OFFLINE_PRETRAIN_DIR
    log_dir = f"{configs.OFFLINE_PRETRAIN_DIR}/logs"
    Path(save_path).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    return [
        "--train_data_dir", configs.TRAIN_DATA_DIR,
        "--save_path", save_path,
        "--log_dir", log_dir,
        "--seed", "42",
        "--offline_epochs", str(int(configs.OFFLINE_EPOCHS)),
        "--online_steps", "0",
    ]


def _run_trainer(argv: list[str], patch_name: str | None = None) -> None:
    _apply_patch(patch_name)
    from train.o2o_iql import trainer

    original_argv = sys.argv
    sys.argv = ["train.o2o_iql.trainer", *argv]
    try:
        trainer.main()
    finally:
        sys.argv = original_argv


def _finalize_pretrain_ckpt() -> None:
    src = Path(configs.OFFLINE_PRETRAIN_DIR) / "offline_iql_final.pt"
    dst = Path(configs.SHARED_CKPT)
    if not src.exists():
        raise FileNotFoundError(f"Expected pretraining output at {src}, not found.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Copy (not symlink, for portability on Windows).
    import shutil
    shutil.copy2(src, dst)
    print(f"  Shared offline checkpoint saved -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one O2O IQL ablation variant.")
    parser.add_argument("--variant", type=str, default=None, choices=list(configs.ABLATIONS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--pretrain_only",
        action="store_true",
        help="Run the shared offline pretraining pass and save SHARED_CKPT, then exit.",
    )
    args = parser.parse_args()

    if args.pretrain_only:
        print(f"=== Offline pretraining (shared ckpt) ===")
        _run_trainer(_pretrain_argv(), patch_name=None)
        _finalize_pretrain_ckpt()
        return

    if args.variant is None or args.seed is None:
        parser.error("--variant and --seed are required unless --pretrain_only is given.")

    argv, patch_name = _variant_argv(args.variant, args.seed)
    print(f"=== Ablation run: variant={args.variant}  seed={args.seed} ===")
    print(f"  argv: {' '.join(argv)}")
    _run_trainer(argv, patch_name=patch_name)


if __name__ == "__main__":
    main()
