import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import combo_log_bias
import search
from source_identity import source_identity
import test_search_integration as fixtures


class ReplayIntegrationTests(unittest.TestCase):
    def test_completed_legacy_grid_replays_without_optimizer_fits_or_new_scores(self):
        legacy_root = ROOT.parent / 'staged_combo_search_20260916_v2'
        spec = importlib.util.spec_from_file_location('legacy_search_for_test', legacy_root / 'search.py')
        legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = fixtures.SearchIntegrationTests().fixture(root)
            config['meta_source_sha256'] = source_identity(legacy_root)[0]
            config.pop('_runtime_config_sha256', None)
            config_path = root / 'producer_config.json'
            config_path.write_text(json.dumps(config))
            config['_runtime_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
            origin = root / 'legacy_run'
            def zeros(probability, y):
                return np.zeros(probability.shape[1]), {'synthetic': True}
            with patch.object(combo_log_bias, 'fit_c10_log_bias', zeros):
                prior = legacy.run_search(config, origin)
            recovery_config = {**config, 'meta_source_sha256': source_identity(ROOT)[0],
                               '_runtime_config_sha256': 'recovery-config',
                               'completed_replay': {'source_root': str(legacy_root), 'run_root': str(origin),
                                                    'config_path': str(config_path),
                                                    'source_sha256': config['meta_source_sha256'],
                                                    'config_sha256': config['_runtime_config_sha256']}}
            target = root / 'recovered'
            with patch.object(combo_log_bias, 'fit_c10_log_bias', side_effect=AssertionError('optimizer must not refit')):
                result = search.run_search(recovery_config, target)
            self.assertTrue(result['completed_score_replay'])
            self.assertEqual(result['declared_score_rows'], 6241)
            self.assertEqual(result['best_recipe_id'], prior['best_recipe_id'])
            self.assertEqual(result['best_inner_macro_f1'], prior['best_inner_macro_f1'])
            self.assertEqual(result['score_producer_source_sha256'], config['meta_source_sha256'])
            self.assertNotEqual(result['score_producer_source_sha256'], result['meta_source_sha256'])
            for phase in ('C0', 'C1', 'C2'):
                for suffix in ('.jsonl', '.IDENTITY.json', '.COMPLETE.json'):
                    self.assertEqual((origin / 'ledgers' / (phase + suffix)).read_bytes(),
                                     (target / 'ledgers' / (phase + suffix)).read_bytes())


if __name__ == '__main__': unittest.main()
