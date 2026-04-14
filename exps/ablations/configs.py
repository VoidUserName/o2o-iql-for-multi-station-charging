"""Ablation study configuration for O2O IQL.

Each variant maps to a set of CLI overrides for train/o2o_iql/trainer.py plus an
optional runtime patch name consumed by run_variant.py.

Patch names:
    None                  -> no patching
    "actor_stochastic"    -> replace DiscreteIQLAgent.act_ucb with
                             self.act(obs, mask, deterministic=False)
"""
from __future__ import annotations

ROOT = "exps/ablations"
SHARED_CKPT = f"{ROOT}/shared_offline_ckpt.pt"
OFFLINE_PRETRAIN_DIR = f"{ROOT}/_offline_pretrain"

TRAIN_DATA_DIR = "data/train_dataset/bias"
EVAL_DATA_ROOT = "data/train_dataset"
ONLINE_STEPS = 200_000
OFFLINE_EPOCHS = 100
SEEDS = [42, 123, 2024]
SPLITS = ["bias", "extreme", "idle", "normal"]

ABLATIONS: dict[str, dict] = {
    "full_o2o_iql": {
        "use_pretrained": True,
        "patch": None,
        "cli": {},
    },
    "no_offline": {
        "use_pretrained": False,
        "patch": None,
        "cli": {"offline_epochs": 0},
    },
    "no_dual_buffer": {
        "use_pretrained": True,
        "patch": None,
        "cli": {
            "online_sample_prob": 1.0,
            "min_online_samples": 1,
            "priority_refresh_freq": 999_999_999,
        },
    },
    "no_density_priority": {
        "use_pretrained": True,
        "patch": None,
        "cli": {
            "priority_uniform_floor": 1.0,
            "priority_refresh_freq": 999_999_999,
        },
    },
    "no_anneal": {
        "use_pretrained": True,
        "patch": None,
        "cli": {"anneal_steps": 1},
    },
    "no_ucb": {
        "use_pretrained": True,
        "patch": "actor_stochastic",
        "cli": {},
    },
}


def variant_run_dir(variant: str, seed: int) -> str:
    return f"{ROOT}/runs/{variant}/seed{seed}"


def variant_save_path(variant: str, seed: int) -> str:
    return f"{variant_run_dir(variant, seed)}/ckpt"


def variant_log_dir(variant: str, seed: int) -> str:
    return f"{variant_run_dir(variant, seed)}/logs"


def variant_result_dir(variant: str) -> str:
    return f"{ROOT}/results/{variant}"
