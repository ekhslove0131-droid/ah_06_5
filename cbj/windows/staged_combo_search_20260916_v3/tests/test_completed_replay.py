import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completed_replay import adopt_completed_search
from source_identity import source_identity


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CompletedReplayTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'; source.mkdir()
        (source / 'SOURCE_ALLOWLIST.txt').write_text('engine.py\n')
        (source / 'engine.py').write_text('# original producer\n')
        source_sha = source_identity(source)[0]
        config_path = source / 'runtime.json'
        config_path.write_text(json.dumps({'meta_source_sha256': source_sha,
                                          'evidence_sha256': 'e', 'grid_manifest_sha256': 'g'}))
        origin = root / 'origin'; (origin / 'ledgers').mkdir(parents=True)
        (origin / 'bias_cache').mkdir()
        identity = {'schema_version': 'COMBO_SEARCH_IDENTITY_V1', 'evidence_sha256': 'e',
                    'rows': 2, 'candidate_count': 124, 'grid_manifest_sha256': 'g',
                    'meta_source_sha256': 'new', 'runtime_config_sha256': 'newconfig'}
        producer = {**identity, 'meta_source_sha256': source_sha,
                    'runtime_config_sha256': sha(config_path)}
        for phase, count in [('C0', 3721), ('C1', 10), ('C2', 10)]:
            path = origin / 'ledgers' / f'{phase}.jsonl'
            path.write_text(''.join(json.dumps({'recipe_id': f'{phase}-{i}', 'declared_index': i}) + '\n' for i in range(count)))
            header = {**producer, 'round': phase}
            if phase != 'C0': header['parent_ids'] = ['one_parent']
            path.with_suffix('.IDENTITY.json').write_text(json.dumps(header))
            path.with_suffix('.COMPLETE.json').write_text(json.dumps({'declared_rows': count, 'ledger_sha256': sha(path)}))
        config = {'completed_replay': {'source_root': str(source), 'run_root': str(origin),
                                     'config_path': str(config_path), 'source_sha256': source_sha,
                                     'config_sha256': sha(config_path)}}
        return config, identity, producer, origin

    def test_copy_preserves_producer_identity_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, identity, producer, origin = self.fixture(root)
            target = root / 'newrun'
            actual, receipt = adopt_completed_search(config, target, identity)
            self.assertTrue((target / 'ledgers/C2.jsonl').is_file())
            self.assertEqual(actual, producer)
            self.assertEqual(receipt['score_producer_source_sha256'], producer['meta_source_sha256'])
            self.assertEqual(receipt['recovery_source_sha256'], 'new')
            self.assertEqual(sha(target / 'ledgers/C2.jsonl'), sha(origin / 'ledgers/C2.jsonl'))
            self.assertEqual(adopt_completed_search(config, target, identity), (actual, receipt))

    def test_tampered_origin_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, identity, _, origin = self.fixture(root)
            path = origin / 'ledgers/C2.jsonl'; path.write_bytes(path.read_bytes() + b'bad')
            with self.assertRaises(ValueError): adopt_completed_search(config, root / 'newrun', identity)
            self.assertFalse((root / 'newrun/ledgers/C0.jsonl').exists())

    def test_mismatched_destination_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, identity, _, _ = self.fixture(root)
            target = root / 'newrun'; adopt_completed_search(config, target, identity)
            path = target / 'ledgers/C0.jsonl'
            self.assertTrue(path.is_file())
            path.write_bytes(b'preserve mismatch')
            with self.assertRaises(ValueError): adopt_completed_search(config, target, identity)
            self.assertEqual(path.read_bytes(), b'preserve mismatch')


if __name__ == '__main__': unittest.main()
