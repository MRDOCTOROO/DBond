"""One-shot, event-backed pod launcher for the v3 k5 ranking ensemble."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


REPO = Path(__file__).resolve().parents[2]
OPS = Path('/mnt/pvc/graphtrans/four_state_k5_seed_ensemble_v3_ops')
PREFIX = REPO / 'checkpoints/graph_transform/four_state_k5_seed_ensemble_v3_batch'
RUNNER = REPO / 'graph_transform/scripts/ensemble_k5_same_fold_v3.py'
PROTOCOL = REPO / 'experiments/k5_seed_ensemble_ranking_v3.md'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def current_revision() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def launch(device: str, batch_size: int) -> None:
    if not sys.platform.startswith('linux') or not str(REPO).startswith('/mnt/pvc/graphtrans/'):
        raise RuntimeError('Launcher is restricted to the authorized pod')
    if not (RUNNER.is_file() and PROTOCOL.is_file() and (OPS / 'preflight.json').is_file()):
        raise RuntimeError('Runner, protocol, and completed pod preflight are required')
    if (OPS / 'launch.json').exists() or (OPS / 'terminal.json').exists():
        raise RuntimeError('An existing launch or terminal receipt forbids a duplicate submission')
    preflight = read_json(OPS / 'preflight.json')
    if preflight.get('status') != 'verified' or len(preflight.get('folds_checked', [])) != 5:
        raise RuntimeError('All five source/split/checkpoint preflight checks must pass')
    if batch_size != 512 or not device.startswith('cuda:'):
        raise RuntimeError('Use one GPU and the frozen evaluation batch size 512')
    OPS.mkdir(parents=True, exist_ok=True)
    PREFIX.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d_%H%M%S')
    root = PREFIX / stamp
    root.mkdir(exist_ok=False)
    claim = OPS / 'launch.claim'
    with claim.open('x', encoding='utf-8') as stream:
        stream.write(str(root) + '\n')
    receipt = {
        'root': str(root), 'ops': str(OPS), 'gitrev': current_revision(),
        'submitted_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'device': device, 'batch_size': batch_size,
        'runner': str(RUNNER), 'runner_sha256': sha(RUNNER),
        'protocol': str(PROTOCOL), 'protocol_sha256': sha(PROTOCOL),
        'preflight': str(OPS / 'preflight.json'),
        'preflight_sha256': sha(OPS / 'preflight.json'),
        'member_order': ['seed42', 'seed43', 'seed45'],
        'operation': 'same-fold fixed equal-probability inference; no training',
    }
    atomic_json(OPS / 'launch.json', receipt)
    with (OPS / 'worker.log').open('ab') as output:
        worker = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), 'worker', '--root', str(root)],
            cwd=REPO, stdout=output, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, start_new_session=True,
        )
    receipt['worker_pid'] = worker.pid
    atomic_json(OPS / 'launch.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def worker(root: Path) -> None:
    receipt = read_json(OPS / 'launch.json')
    if root != Path(receipt['root']) or (OPS / 'terminal.json').exists():
        raise RuntimeError('Worker root mismatch or terminal already exists')
    status, error = 'failed', None
    try:
        command = [sys.executable, str(RUNNER), '--mode', 'run', '--out', str(OPS),
                   '--device', receipt['device'], '--batch-size', str(receipt['batch_size'])]
        with (OPS / 'inference.log').open('ab') as output:
            code = subprocess.call(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
        if code != 0:
            raise RuntimeError(f'Fixed inference runner exited with code {code}')
        audit = read_json(OPS / 'audit_receipt.json')
        result = read_json(OPS / 'result.json')
        if audit.get('status') != 'passed' or result.get('status') != 'completed':
            raise RuntimeError('Audit or result does not report successful completion')
        status = 'completed'
    except Exception as exc:
        error = str(exc)
        traceback.print_exc()
    terminal = {
        'root': str(root), 'gitrev': receipt['gitrev'], 'status': status,
        'finished_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'result': str(OPS / 'result.json') if status == 'completed' else None,
        'report': str(OPS / 'report.md') if status == 'completed' else None,
        'audit': str(OPS / 'audit_receipt.json') if status == 'completed' else None,
        'error': error,
    }
    atomic_json(OPS / 'terminal.json', terminal)
    print(json.dumps(terminal, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('launch', 'worker'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--root', type=Path)
    args = parser.parse_args()
    if args.action == 'launch':
        launch(args.device, args.batch_size)
    elif args.root is None:
        parser.error('--root is required for worker')
    else:
        worker(args.root)
