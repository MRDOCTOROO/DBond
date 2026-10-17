#!/usr/bin/env python3
"""Run ordered architecture arms and summarize only a complete, successful batch.

The first --configs entry is the control. --run_root must be new or empty.
Each arm has a manifest-bound output directory, never a "latest" lookup.
Run from the project environment on the training pod.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from summarize_architecture_batch import FOLDS, atomic_json, load_fold_metrics, sha256, utc_now

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "graph_transform" / "scripts"


def git_revision(repo: Path) -> str | None:
    """Read HEAD without modifying Git state or requiring a git executable."""
    git_dir = repo / ".git"
    if git_dir.is_file():
        text = git_dir.read_text(encoding="utf-8").strip()
        if not text.startswith("gitdir: "):
            return None
        git_dir = (repo / text[8:]).resolve()
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        common = git_dir
        if (git_dir / "commondir").exists():
            common = (git_dir / (git_dir / "commondir").read_text().strip()).resolve()
        for directory in (git_dir, common):
            if (directory / ref).is_file():
                return (directory / ref).read_text().strip()
        packed = common / "packed-refs"
        if packed.exists():
            for line in packed.read_text().splitlines():
                fields = line.split()
                if len(fields) == 2 and fields[1] == ref:
                    return fields[0]
    except OSError:
        pass
    return None


def stop_child(proc: subprocess.Popen) -> None:
    """Stop the training process group, including its fold workers, on interruption."""
    if proc.poll() is not None:
        return
    if os.name == "posix":
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
        proc.wait()


def run_logged(command: list[str], log_path: Path, status: dict, status_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=REPO_ROOT, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1,
                                start_new_session=os.name == "posix")
        try:
            status["child_pid"] = proc.pid
            status["updated_at"] = utc_now()
            atomic_json(status_path, status)
            assert proc.stdout is not None
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return proc.wait()
        finally:
            stop_child(proc)
            if proc.stdout is not None:
                proc.stdout.close()
            status["child_pid"] = None
            status["updated_at"] = utc_now()
            atomic_json(status_path, status)


def execute(args: argparse.Namespace) -> int:
    root = Path(args.run_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError(f"Refusing to overwrite nonempty run_root: {root}")
    # Exclusive creation prevents two dispatchers claiming the same empty root.
    with (root / ".batch.lock").open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    status_path = root / "status.json"
    status = {"schema_version": 1, "status": "running", "phase": "preflight",
              "pid": os.getpid(), "child_pid": None, "root": str(root),
              "current_arm": None, "config": None, "gitrev": git_revision(REPO_ROOT),
              "started_at": utc_now(), "updated_at": utc_now(), "finished_at": None,
              "arms": [], "error": None}
    atomic_json(status_path, status)
    try:
        configs = [Path(value).expanduser().resolve() for value in args.configs]
        if not configs or len(set(configs)) != len(configs):
            raise ValueError("Supply one or more distinct configs, first = control")
        data = Path(args.fold_data_dir).expanduser().resolve()
        if not data.is_dir():
            raise FileNotFoundError(f"fold_data_dir does not exist: {data}")
        gpu_ids = [int(value.strip()) for value in args.gpus.split(",") if value.strip()]
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids) or any(gpu not in range(5) for gpu in gpu_ids):
            raise ValueError("gpus must be distinct values from 0,1,2,3,4")
        gpus = ",".join(map(str, gpu_ids))
        arms = []
        for index, config in enumerate(configs, 1):
            if not config.is_file():
                raise FileNotFoundError(f"Config missing: {config}")
            slug = re.sub(r"[^A-Za-z0-9_-]", "_", config.stem)
            name = f"{index:02d}_{slug}"
            arm_root = root / "arms" / name
            arm_root.mkdir(parents=True)
            snapshot = arm_root / "config.yaml"
            snapshot.write_bytes(config.read_bytes())
            arms.append({"name": name, "config": str(config),
                         "config_snapshot": str(snapshot), "config_sha256": sha256(snapshot),
                         "cv_root": str(arm_root / "cv"),
                         "work_root": str(arm_root / "logs"),
                         "run_log": str(arm_root / "run.log"),
                         "metrics_csv": str(arm_root / "cv" / "5fold_metrics.csv")})
        manifest = {"schema_version": 1, "created_at": utc_now(), "root": str(root),
                    "repo_root": str(REPO_ROOT), "gitrev": status["gitrev"],
                    "gitrev_note": "HEAD only; uncommitted changes are not represented by the revision.",
                    "fold_data_dir": str(data), "gpus": gpu_ids, "folds": list(FOLDS),
                    "control": arms[0]["name"], "arms": arms,
                    "script_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path)
                                      for path in (Path(__file__).resolve(),
                                                   SCRIPTS / "train_5fold_parallel.py",
                                                   SCRIPTS / "train_5fold.py",
                                                   SCRIPTS / "summarize_architecture_batch.py")}}
        manifest_path = root / "manifest.json"
        atomic_json(manifest_path, manifest)
        status["manifest"] = str(manifest_path)
        status["arms"] = [{"name": arm["name"], "status": "pending", "started_at": None,
                           "finished_at": None, "returncode": None} for arm in arms]
        for arm, arm_status in zip(arms, status["arms"]):
            status.update(phase="training", current_arm=arm["name"], config=arm["config"],
                          updated_at=utc_now())
            arm_status.update(status="running", started_at=utc_now())
            atomic_json(status_path, status)
            command = [sys.executable, str(SCRIPTS / "train_5fold_parallel.py"),
                       "--config", arm["config_snapshot"], "--fold_data_dir", str(data),
                       "--gpus", gpus, "--cv_root", arm["cv_root"],
                       "--work_root", arm["work_root"]]
            print(f"[batch] starting {arm['name']} -> {arm['cv_root']}", flush=True)
            rc = run_logged(command, Path(arm["run_log"]), status, status_path)
            arm_status["returncode"] = rc
            if rc != 0:
                raise RuntimeError(f"Arm {arm['name']} failed (exit {rc}); see {arm['run_log']}")
            load_fold_metrics(Path(arm["metrics_csv"]))
            arm_status.update(status="completed", finished_at=utc_now())
            status["updated_at"] = utc_now()
            atomic_json(status_path, status)
        status.update(phase="summarizing", current_arm=None, config=None, updated_at=utc_now())
        atomic_json(status_path, status)
        rc = run_logged([sys.executable, str(SCRIPTS / "summarize_architecture_batch.py"),
                         "--manifest", str(manifest_path)], root / "summary.log", status, status_path)
        if rc != 0:
            raise RuntimeError(f"Summary failed (exit {rc}); see {root / 'summary.log'}")
        report_path = root / "summary" / "summary.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if [source["arm"] for source in report["sources"]] != [arm["name"] for arm in arms]:
            raise ValueError("Summary output does not match this batch manifest")
        status.update(status="completed", phase="completed", summary=str(report_path),
                      finished_at=utc_now(), updated_at=utc_now())
        atomic_json(status_path, status)
        print(f"[batch] completed: {report_path}", flush=True)
        return 0
    except BaseException as error:
        for arm_status in status["arms"]:
            if arm_status["status"] == "running":
                arm_status.update(status="failed", finished_at=utc_now())
        status.update(status="failed", error=f"{type(error).__name__}: {error}",
                      finished_at=utc_now(), updated_at=utc_now())
        atomic_json(status_path, status)
        print(f"[batch] failed: {status['error']}", file=sys.stderr, flush=True)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True, help="Ordered YAMLs; first is control")
    parser.add_argument("--fold_data_dir", required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--run_root", required=True, help="New or empty batch directory")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        return execute(args)
    except Exception as error:
        print(f"[batch] {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
