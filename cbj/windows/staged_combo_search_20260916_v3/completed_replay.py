"""Adopt immutable completed score records without relabeling their producer."""

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re

from source_identity import source_identity


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write_once(path, payload):
    path = Path(path)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError('replay destination differs; preserving existing file')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.partial')
    with temp.open('wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)


def adopt_completed_search(config, run_root, identity):
    spec = config.get('completed_replay')
    if not spec:
        return identity, None
    origin, target = Path(spec['run_root']), Path(run_root)
    if origin.resolve() == target.resolve():
        raise ValueError('replay must preserve a separate origin run')
    lock_path = origin / '.runner.lock'
    if lock_path.is_file():
        with lock_path.open('rb') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError('origin search is still running') from error
            fcntl.flock(lock, fcntl.LOCK_UN)
    if source_identity(spec['source_root'])[0] != spec['source_sha256']:
        raise ValueError('origin source identity changed')
    if _sha(spec['config_path']) != spec['config_sha256']:
        raise ValueError('origin config identity changed')
    original_config = _read(spec['config_path'])
    if original_config['meta_source_sha256'] != spec['source_sha256']:
        raise ValueError('origin source/config binding differs')
    for key in ('evidence_sha256', 'grid_manifest_sha256'):
        if original_config[key] != identity[key]:
            raise ValueError('replay inputs differ from score producer')
    producer = {**identity, 'meta_source_sha256': spec['source_sha256'],
                'runtime_config_sha256': spec['config_sha256']}
    files, counts = {}, {}
    for phase in ('C0', 'C1', 'C2'):
        path = origin / 'ledgers' / (phase + '.jsonl')
        header_path, complete_path = path.with_suffix('.IDENTITY.json'), path.with_suffix('.COMPLETE.json')
        header, complete = _read(header_path), _read(complete_path)
        if {k: v for k, v in header.items() if k != 'parent_ids'} != {**producer, 'round': phase}:
            raise ValueError('origin ledger identity differs')
        parents = header.get('parent_ids', [])
        if phase != 'C0' and (not 1 <= len(parents) <= 6 or len(set(parents)) != len(parents)):
            raise ValueError('origin parent membership is invalid')
        expected_count = 3721 if phase == 'C0' else math.comb(len(parents) + 3, 4) * 10
        if _sha(path) != complete['ledger_sha256']:
            raise ValueError('origin score ledger hash changed')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if (len(rows) != expected_count or complete['declared_rows'] != expected_count
                or [row['declared_index'] for row in rows] != list(range(expected_count))
                or len({row['recipe_id'] for row in rows}) != expected_count):
            raise ValueError('origin score grid is incomplete')
        counts[phase] = expected_count
        for file in (path, header_path, complete_path):
            files[str(file.relative_to(origin))] = _sha(file)
    for file in sorted((origin / 'bias_cache').glob('*.json')):
        if not re.fullmatch(r'[0-9a-f]{64}\.json', file.name):
            raise ValueError('unexpected origin bias cache file')
        files[str(file.relative_to(origin))] = _sha(file)
    receipt = {
        'schema_version': 'COMPLETED_SCORE_REPLAY_V1', 'origin_run_root': str(origin.resolve()),
        'score_producer_source_sha256': spec['source_sha256'],
        'score_producer_config_sha256': spec['config_sha256'],
        'recovery_source_sha256': identity['meta_source_sha256'],
        'recovery_config_sha256': identity['runtime_config_sha256'],
        'declared_rows_by_round': counts, 'file_sha256': files,
        'original_failure_sha256': _sha(origin / 'FAILURE.json') if (origin / 'FAILURE.json').is_file() else None,
        'score_rows_recomputed': False, 'producer_identities_rewritten': False,
    }
    # Finish all origin checks before copying any artifact. Never relabel the
    # original ledger's source/config hashes as this recovery implementation.
    for relative, digest in files.items():
        payload = (origin / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError('origin artifact changed during adoption')
        _write_once(target / relative, payload)
    _write_once(target / 'ORIGIN.json', (json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode())
    return producer, receipt
