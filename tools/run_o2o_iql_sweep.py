"""Standalone sweep runner for full O2O IQL experiments.

This script keeps the ablation code untouched and drives train/o2o_iql/trainer.py
directly. It first builds a shared offline checkpoint from data/offline_dataset,
then reuses it for 4 scenario x 5 seed online runs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = ["bias", "extreme", "idle", "normal"]
DEFAULT_SEEDS = [42, 123, 2024, 3407, 3408]
DEFAULT_OFFLINE_DATA_DIR = ROOT / "data" / "offline_dataset"
DEFAULT_TRAIN_DATA_ROOT = ROOT / "data" / "train_dataset"
DEFAULT_RUN_ROOT = ROOT / "runs" / "o2o_iql"
DEFAULT_CACHE_PATH = DEFAULT_RUN_ROOT / "_cache" / "offline_dataset.npz"
DEFAULT_SHARED_PRETRAIN_DIR = DEFAULT_RUN_ROOT / "_shared_offline_pretrain"
DEFAULT_SHARED_CKPT = DEFAULT_RUN_ROOT / "_shared_offline_ckpt.pt"


@dataclass(slots=True)
class HyperParams:
    demand_dir: str = "data/offline_dataset/demand"
    solution_dir: str = "data/offline_dataset/solutions"
    train_data_root: str = "data/train_dataset"
    n_bins: int = 21
    max_queue_len: int = 10
    invalid_action_penalty: float = 0.0
    offline_epochs: int = 100
    online_steps: int = 500_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    discount: float = 0.99
    expectile: float = 0.7
    temperature: float = 3.0
    target_update_rate: float = 5e-3
    exp_adv_max: float = 100.0
    hidden_dim: int = 256
    online_buffer_size: int = 20_000
    online_sample_prob: float = 0.6
    min_online_samples: int = 2_000
    updates_per_step: int = 1
    start_training_after: int = 1
    priority_refresh_freq: int = 5_000
    priority_model_steps: int = 100
    priority_batch_size: int = 512
    priority_model_lr: float = 1e-3
    priority_uniform_floor: float = 0.05
    priority_temperature: float = 1.0
    priority_max_ratio: float = 50.0
    n_eval_episodes: int = 10
    eval_freq: int = 50_000
    checkpoint_freq: int = 50_000
    log_interval: int = 5_000
    anneal_steps: int = 0
    device: str = "auto"
    offline_limit_episodes: int = 0
    train_limit_episodes: int = 0


@dataclass(slots=True)
class RunSpec:
    kind: str
    scenario: str
    seed: int
    save_path: str
    log_dir: str
    console_log: str
    manifest_path: str
    metrics_path: str
    command: list[str]
    started_at: str
    finished_at: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    status: str = "pending"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _cmd_to_string(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    return shlex.join(cmd)


def _git_info() -> dict[str, str]:
    def _run(args: list[str]) -> str:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=True,
            )
        except Exception:
            return "unknown"
        return out.stdout.strip() or "unknown"

    return {
        "commit": _run(["rev-parse", "HEAD"]),
        "branch": _run(["branch", "--show-current"]),
        "status": _run(["status", "--short"]),
    }


def _build_trainer_command(
    *,
    hp: HyperParams,
    save_path: Path,
    log_dir: Path,
    train_data_dir: Path,
    eval_data_dir: Path,
    offline_dataset_cache: Path,
    seed: int,
    pretrained_checkpoint: Path | None,
    online_steps: int,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "train.o2o_iql.trainer",
        "--demand_dir",
        hp.demand_dir,
        "--solution_dir",
        hp.solution_dir,
        "--train_data_dir",
        str(train_data_dir),
        "--eval_data_dir",
        str(eval_data_dir),
        "--offline_dataset_cache",
        str(offline_dataset_cache),
        "--save_path",
        str(save_path),
        "--log_dir",
        str(log_dir),
        "--seed",
        str(int(seed)),
        "--n_bins",
        str(int(hp.n_bins)),
        "--max_queue_len",
        str(int(hp.max_queue_len)),
        "--invalid_action_penalty",
        str(float(hp.invalid_action_penalty)),
        "--offline_epochs",
        str(int(hp.offline_epochs)),
        "--online_steps",
        str(int(online_steps)),
        "--batch_size",
        str(int(hp.batch_size)),
        "--learning_rate",
        str(float(hp.learning_rate)),
        "--discount",
        str(float(hp.discount)),
        "--expectile",
        str(float(hp.expectile)),
        "--temperature",
        str(float(hp.temperature)),
        "--target_update_rate",
        str(float(hp.target_update_rate)),
        "--exp_adv_max",
        str(float(hp.exp_adv_max)),
        "--hidden_dim",
        str(int(hp.hidden_dim)),
        "--online_buffer_size",
        str(int(hp.online_buffer_size)),
        "--online_sample_prob",
        str(float(hp.online_sample_prob)),
        "--min_online_samples",
        str(int(hp.min_online_samples)),
        "--updates_per_step",
        str(int(hp.updates_per_step)),
        "--start_training_after",
        str(int(hp.start_training_after)),
        "--priority_refresh_freq",
        str(int(hp.priority_refresh_freq)),
        "--priority_model_steps",
        str(int(hp.priority_model_steps)),
        "--priority_batch_size",
        str(int(hp.priority_batch_size)),
        "--priority_model_lr",
        str(float(hp.priority_model_lr)),
        "--priority_uniform_floor",
        str(float(hp.priority_uniform_floor)),
        "--priority_temperature",
        str(float(hp.priority_temperature)),
        "--priority_max_ratio",
        str(float(hp.priority_max_ratio)),
        "--n_eval_episodes",
        str(int(hp.n_eval_episodes)),
        "--eval_freq",
        str(int(hp.eval_freq)),
        "--checkpoint_freq",
        str(int(hp.checkpoint_freq)),
        "--log_interval",
        str(int(hp.log_interval)),
        "--anneal_steps",
        str(int(hp.anneal_steps)),
        "--device",
        hp.device,
        "--offline_limit_episodes",
        str(int(hp.offline_limit_episodes)),
        "--train_limit_episodes",
        str(int(hp.train_limit_episodes)),
    ]
    if pretrained_checkpoint is not None:
        cmd.extend(["--pretrained_checkpoint", str(pretrained_checkpoint)])
    return cmd


def _tee_process(
    cmd: list[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
    header: str,
    mirror_stdout: bool = True,
) -> int:
    _ensure_parent(log_path)
    with log_path.open("w", encoding="utf-8", newline="") as log_file:
        log_file.write(f"{header}\n")
        log_file.write(f"$ {_cmd_to_string(cmd)}\n")
        log_file.flush()
        if mirror_stdout:
            print(header, flush=True)
            print(f"$ {_cmd_to_string(cmd)}", flush=True)

        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if mirror_stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                log_file.write(line)
                log_file.flush()
            return_code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                return_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                return_code = proc.wait()
            raise
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
    return int(return_code)


def _make_run_spec(
    *,
    kind: str,
    scenario: str,
    seed: int,
    save_path: Path,
    log_dir: Path,
    console_log: Path,
    manifest_path: Path,
    command: list[str],
    metrics_path: Path,
) -> RunSpec:
    started_at = _utc_now()
    return RunSpec(
        kind=kind,
        scenario=scenario,
        seed=int(seed),
        save_path=str(save_path),
        log_dir=str(log_dir),
        console_log=str(console_log),
        manifest_path=str(manifest_path),
        metrics_path=str(metrics_path),
        command=command,
        started_at=started_at,
    )


def _finalize_spec(spec: RunSpec, *, exit_code: int) -> None:
    spec.exit_code = int(exit_code)
    spec.status = "succeeded" if exit_code == 0 else "failed"
    spec.finished_at = _utc_now()
    start = datetime.fromisoformat(spec.started_at)
    finish = datetime.fromisoformat(spec.finished_at)
    spec.duration_seconds = (finish - start).total_seconds()


def _spec_payload(
    spec: RunSpec,
    *,
    hp: HyperParams,
    git: dict[str, str],
    offline_dataset_cache: Path,
    train_data_dir: Path,
    eval_data_dir: Path,
    pretrained_checkpoint: str | None,
) -> dict[str, Any]:
    payload = asdict(spec)
    payload.update(
        {
            "command_string": _cmd_to_string(spec.command),
            "hyperparameters": asdict(hp),
            "git": git,
            "offline_dataset_cache": str(offline_dataset_cache),
            "train_data_dir": str(train_data_dir),
            "eval_data_dir": str(eval_data_dir),
            "pretrained_checkpoint": pretrained_checkpoint,
            "python": sys.executable,
            "platform": platform.platform(),
            "cwd": str(ROOT),
        }
    )
    return payload


def _prepare_dirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _copy_shared_checkpoint(pretrain_dir: Path, shared_ckpt: Path) -> None:
    final_ckpt = pretrain_dir / "offline_iql_final.pt"
    if not final_ckpt.exists():
        raise FileNotFoundError(f"Expected pretrain checkpoint at {final_ckpt}, not found.")
    _ensure_parent(shared_ckpt)
    shutil.copy2(final_ckpt, shared_ckpt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone full O2O IQL sweep runner.")
    parser.add_argument("--scenarios", nargs="*", default=DEFAULT_SCENARIOS)
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--run_root", type=str, default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--offline_dataset_root", type=str, default=str(DEFAULT_OFFLINE_DATA_DIR))
    parser.add_argument("--train_data_root", type=str, default=str(DEFAULT_TRAIN_DATA_ROOT))
    parser.add_argument("--offline_cache", type=str, default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--shared_pretrain_dir", type=str, default=str(DEFAULT_SHARED_PRETRAIN_DIR))
    parser.add_argument("--shared_ckpt", type=str, default=str(DEFAULT_SHARED_CKPT))
    parser.add_argument("--skip_pretrain", action="store_true")
    parser.add_argument("--pretrain_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_queue_len", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=5_000)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--checkpoint_freq", type=int, default=50_000)
    parser.add_argument("--online_steps", type=int, default=500_000)
    parser.add_argument("--online_buffer_size", type=int, default=20_000)
    parser.add_argument("--offline_epochs", type=int, default=100)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Parallel online runs. 1 = original serial behavior.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help=(
            "Comma-separated CUDA device ids to shard parallel workers across, e.g. '0,1'. "
            "Empty = do not set CUDA_VISIBLE_DEVICES (all workers share current visibility)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hp = HyperParams(
        demand_dir=str(Path(args.offline_dataset_root) / "demand"),
        solution_dir=str(Path(args.offline_dataset_root) / "solutions"),
        train_data_root=str(Path(args.train_data_root)),
        max_queue_len=int(args.max_queue_len),
        offline_epochs=int(args.offline_epochs),
        online_steps=int(args.online_steps),
        online_buffer_size=int(args.online_buffer_size),
        n_eval_episodes=int(args.n_eval_episodes),
        eval_freq=int(args.eval_freq),
        checkpoint_freq=int(args.checkpoint_freq),
        log_interval=int(args.log_interval),
        device=str(args.device),
    )

    run_root = Path(args.run_root)
    shared_pretrain_dir = Path(args.shared_pretrain_dir)
    shared_ckpt = Path(args.shared_ckpt)
    offline_cache = Path(args.offline_cache)
    git = _git_info()

    _prepare_dirs([run_root, shared_pretrain_dir, offline_cache.parent, shared_ckpt.parent])

    sweep_summary_path = run_root / "sweep_manifest.json"
    sweep_summary: dict[str, Any] = {
        "created_at": _utc_now(),
        "python": sys.executable,
        "platform": platform.platform(),
        "cwd": str(ROOT),
        "git": git,
        "scenarios": [str(s) for s in args.scenarios],
        "seeds": [int(s) for s in args.seeds],
        "hyperparameters": asdict(hp),
        "paths": {
            "run_root": str(run_root),
            "shared_pretrain_dir": str(shared_pretrain_dir),
            "shared_ckpt": str(shared_ckpt),
            "offline_cache": str(offline_cache),
        },
        "runs": [],
        "status": "running",
    }
    _write_json(sweep_summary_path, sweep_summary)

    failures: list[dict[str, Any]] = []

    def _record_run(spec: RunSpec) -> None:
        sweep_summary["runs"].append(asdict(spec))
        _write_json(sweep_summary_path, sweep_summary)

    if not args.skip_pretrain:
        pretrain_spec = _make_run_spec(
            kind="shared_pretrain",
            scenario="shared",
            seed=42,
            save_path=shared_pretrain_dir,
            log_dir=shared_pretrain_dir / "logs",
            console_log=shared_pretrain_dir / "console.log",
            manifest_path=shared_pretrain_dir / "manifest.json",
            command=[],
            metrics_path=shared_pretrain_dir / "logs" / "metrics.jsonl",
        )
        train_data_dir = Path(hp.train_data_root) / "normal"
        eval_data_dir = Path(hp.train_data_root) / "normal"
        pretrain_cmd = _build_trainer_command(
            hp=hp,
            save_path=Path(pretrain_spec.save_path),
            log_dir=Path(pretrain_spec.log_dir),
            train_data_dir=train_data_dir,
            eval_data_dir=eval_data_dir,
            offline_dataset_cache=offline_cache,
            seed=pretrain_spec.seed,
            pretrained_checkpoint=None,
            online_steps=0,
        )
        pretrain_spec.command = pretrain_cmd
        if args.dry_run:
            pretrain_spec.status = "dry_run"
            pretrain_spec.finished_at = _utc_now()
            pretrain_spec.duration_seconds = 0.0
            pretrain_spec.exit_code = 0
        _write_json(
            Path(pretrain_spec.manifest_path),
            _spec_payload(
                pretrain_spec,
                hp=hp,
                git=git,
                offline_dataset_cache=offline_cache,
                train_data_dir=train_data_dir,
                eval_data_dir=eval_data_dir,
                pretrained_checkpoint=None,
            ),
        )
        if args.dry_run:
            print(f"[dry-run] {_cmd_to_string(pretrain_cmd)}", flush=True)
            sweep_summary["runs"].append(asdict(pretrain_spec))
            _write_json(sweep_summary_path, sweep_summary)
        else:
            rc = _tee_process(
                pretrain_cmd,
                log_path=Path(pretrain_spec.console_log),
                cwd=ROOT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                header="=== shared_pretrain | scenario=shared | seed=42 ===",
            )
            _finalize_spec(pretrain_spec, exit_code=rc)
            _write_json(
                Path(pretrain_spec.manifest_path),
                _spec_payload(
                    pretrain_spec,
                    hp=hp,
                    git=git,
                    offline_dataset_cache=offline_cache,
                    train_data_dir=train_data_dir,
                    eval_data_dir=eval_data_dir,
                    pretrained_checkpoint=None,
                ),
            )
            _record_run(pretrain_spec)
            if rc != 0:
                sweep_summary["status"] = "failed"
                sweep_summary["finished_at"] = _utc_now()
                _write_json(sweep_summary_path, sweep_summary)
                raise SystemExit(rc)
            _copy_shared_checkpoint(shared_pretrain_dir, shared_ckpt)
            print(f"Shared checkpoint saved -> {shared_ckpt}", flush=True)
    else:
        if not shared_ckpt.exists():
            raise FileNotFoundError(
                f"--skip_pretrain was set but shared checkpoint is missing: {shared_ckpt}"
            )

    if args.pretrain_only:
        sweep_summary["status"] = "dry_run" if args.dry_run else "succeeded"
        sweep_summary["finished_at"] = _utc_now()
        _write_json(sweep_summary_path, sweep_summary)
        return

    total_runs = len(args.scenarios) * len(args.seeds)
    max_workers = max(1, int(args.max_workers))
    gpu_ids = [g.strip() for g in str(args.gpu_ids).split(",") if g.strip()]
    print(
        f"=== Running full O2O IQL sweep: {len(args.scenarios)} scenarios x {len(args.seeds)} seeds = {total_runs} runs ===",
        flush=True,
    )
    print(f"Shared checkpoint: {shared_ckpt}", flush=True)
    print(f"Offline cache: {offline_cache}", flush=True)
    print(
        f"Parallelism: max_workers={max_workers} gpu_ids={gpu_ids or 'inherit'}",
        flush=True,
    )

    @dataclass(slots=True)
    class _PendingRun:
        scenario: str
        seed: int
        run_dir: Path
        manifest_path: Path
        console_log: Path
        train_data_dir: Path
        eval_data_dir: Path
        spec: RunSpec

    pending: list[_PendingRun] = []
    for scenario in args.scenarios:
        train_data_dir = Path(hp.train_data_root) / scenario
        eval_data_dir = train_data_dir
        if not train_data_dir.exists():
            raise FileNotFoundError(f"Scenario directory not found: {train_data_dir}")
        for seed in args.seeds:
            run_dir = run_root / scenario / f"seed{int(seed)}"
            save_path = run_dir / "ckpt"
            log_dir = run_dir / "logs"
            console_log = run_dir / "console.log"
            manifest_path = run_dir / "manifest.json"
            _prepare_dirs([save_path, log_dir, run_dir])
            run_spec = _make_run_spec(
                kind="full_o2o_iql",
                scenario=str(scenario),
                seed=int(seed),
                save_path=save_path,
                log_dir=log_dir,
                console_log=console_log,
                manifest_path=manifest_path,
                command=[],
                metrics_path=log_dir / "metrics.jsonl",
            )
            run_cmd = _build_trainer_command(
                hp=hp,
                save_path=save_path,
                log_dir=log_dir,
                train_data_dir=train_data_dir,
                eval_data_dir=eval_data_dir,
                offline_dataset_cache=offline_cache,
                seed=run_spec.seed,
                pretrained_checkpoint=shared_ckpt,
                online_steps=hp.online_steps,
            )
            run_spec.command = run_cmd
            if args.dry_run:
                run_spec.status = "dry_run"
                run_spec.finished_at = _utc_now()
                run_spec.duration_seconds = 0.0
                run_spec.exit_code = 0
            _write_json(
                manifest_path,
                _spec_payload(
                    run_spec,
                    hp=hp,
                    git=git,
                    offline_dataset_cache=offline_cache,
                    train_data_dir=train_data_dir,
                    eval_data_dir=eval_data_dir,
                    pretrained_checkpoint=str(shared_ckpt),
                ),
            )
            if args.dry_run:
                print(f"[dry-run] {_cmd_to_string(run_cmd)}", flush=True)
                sweep_summary["runs"].append(asdict(run_spec))
                _write_json(sweep_summary_path, sweep_summary)
                continue
            pending.append(
                _PendingRun(
                    scenario=str(scenario),
                    seed=int(seed),
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    console_log=console_log,
                    train_data_dir=train_data_dir,
                    eval_data_dir=eval_data_dir,
                    spec=run_spec,
                )
            )

    summary_lock = threading.Lock()

    def _execute(item: _PendingRun, worker_idx: int) -> int:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = gpu_ids[worker_idx % len(gpu_ids)]
        header = f"=== full_o2o_iql | scenario={item.scenario} | seed={item.seed} ==="
        mirror = max_workers == 1
        if not mirror:
            print(
                f"[start] scenario={item.scenario} seed={item.seed} "
                f"gpu={env.get('CUDA_VISIBLE_DEVICES', 'inherit')} log={item.console_log}",
                flush=True,
            )
        rc = _tee_process(
            item.spec.command,
            log_path=item.console_log,
            cwd=ROOT,
            env=env,
            header=header,
            mirror_stdout=mirror,
        )
        _finalize_spec(item.spec, exit_code=rc)
        _write_json(
            item.manifest_path,
            _spec_payload(
                item.spec,
                hp=hp,
                git=git,
                offline_dataset_cache=offline_cache,
                train_data_dir=item.train_data_dir,
                eval_data_dir=item.eval_data_dir,
                pretrained_checkpoint=str(shared_ckpt),
            ),
        )
        with summary_lock:
            _record_run(item.spec)
            if rc != 0:
                failures.append(
                    {
                        "scenario": item.scenario,
                        "seed": item.seed,
                        "exit_code": int(rc),
                        "run_dir": str(item.run_dir),
                    }
                )
        if not mirror:
            status = "ok" if rc == 0 else f"FAIL({rc})"
            print(
                f"[done]  scenario={item.scenario} seed={item.seed} "
                f"status={status} duration={item.spec.duration_seconds:.1f}s",
                flush=True,
            )
        return rc

    if pending:
        if max_workers == 1:
            for item in pending:
                _execute(item, worker_idx=0)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [
                    pool.submit(_execute, item, idx) for idx, item in enumerate(pending)
                ]
                for fut in as_completed(futures):
                    fut.result()

    sweep_summary["finished_at"] = _utc_now()
    sweep_summary["status"] = "failed" if failures else ("dry_run" if args.dry_run else "succeeded")
    sweep_summary["failures"] = failures
    _write_json(sweep_summary_path, sweep_summary)

    print("\n=== Sweep summary ===", flush=True)
    print(f"total_runs={total_runs}", flush=True)
    print(f"failed={len(failures)}", flush=True)
    if failures:
        for item in failures:
            print(
                f"  - scenario={item['scenario']} seed={item['seed']} exit={item['exit_code']} dir={item['run_dir']}",
                flush=True,
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
