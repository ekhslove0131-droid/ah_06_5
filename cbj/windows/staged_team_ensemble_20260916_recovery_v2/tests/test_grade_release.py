"""Exercise stage orchestration and crash recovery without expensive model fits."""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import grade_runner as gr
from atomic_state import atomic_write_json
from ensemble_grid import MetaCandidate


class GradeReleaseTests(unittest.TestCase):
    def test_csv_precedes_next_grade_and_resume_keeps_completed_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / 'config.json'
            train_path = root / 'train.csv'
            reference_path = root / 'reference.npz'
            train_path.write_text('synthetic input', encoding='utf-8')
            reference_path.write_text('synthetic identities', encoding='utf-8')
            config = {'run_root': str(root), 'target_macro_f1': .6,
                      'train_csv': str(train_path), 'reference_oof': str(reference_path),
                      'test_sha256': '1' * 64, 'sample_submission_sha256': '2' * 64}
            atomic_write_json(config_path, config)
            classes = [f'C{i}' for i in range(26)]
            data = {'ids': np.array(['R0','R1','R2']), 'y_int': np.array([0,1,2]),
                    'groups': np.array(['G0','G1','G2']), 'folds': np.array([0,1,2]),
                    'class_order': classes, 'y_label': np.array(classes[:3])}
            parts = {'outer': [{'fold':0, 'train_indices':[0,1,2]}]}
            report = {'admission_sha256':'a' * 64}
            a0 = [{'candidate_id':'C0'}]
            by_id = {'C0':a0[0], 'C1':{'candidate_id':'C1'}}
            p = np.full((3,26), 1/26.)
            selected = MetaCandidate('R0-test',0,'A1',('C0',)*4,(1.,0.,0.,0.,0.),
                                     'arithmetic',p,.4,'unused')
            selection = {'selected':selected, 'candidate_graph':{'R0-test':selected}, 'manifests':[]}
            test = pd.DataFrame({'ID':['T0','T1']})
            sample = test.assign(SUBCLASS='C0')
            prepared = {'source_sha256':'s'*64,'partition_sha256':gr.canonical_sha256(parts),
                        'admission_sha256':report['admission_sha256'],
                        'train_sha256':gr.sha256_file(train_path),
                        'reference_sha256':gr.sha256_file(reference_path),
                        'runtime_sha256':gr.sha256_file(config_path),
                        'runtime_config_sha256':gr.sha256_file(config_path)}
            atomic_write_json(root/'PREPARED.json', prepared)
            evidence = (np.arange(3),np.arange(3),{'A1':p,'A2':p},p,np.zeros(3,bool),{'C0':p})
            deployments = []
            failures = [True]

            def new_candidates(config, run_root, data, outer, folds, declarations, source):
                if not declarations:
                    return {}, []
                self.assertTrue((root/'grades/A0/COMPLETE.json').is_file())
                self.assertTrue((root/'submissions/submission_grade_A0.csv').is_file())
                if failures[0]:
                    return {}, ['C1']
                return {'C1':p}, []

            def deploy(config, run_root, data, partitions, test, frozen, *args):
                grade = json.loads((root/'STATE.json').read_text())['current_grade']
                self.assertTrue((root/'grades'/grade/'SELECTION.json').is_file())
                deployments.append(grade)
                return p.copy(), np.full((2,26),1/26.), {
                    'refit_plan':{'base_candidate_ids':[]}, 'bias_receipt':{'synthetic':True}}

            with ExitStack() as stack:
                overrides = {
                    'GRADE_ORDER':('A0','A1'), 'GRADE_ADDITIONS':{'A0':(), 'A1':('C1',)},
                    '_source_identity':lambda root: ('s'*64,[]),
                    '_load_training':lambda config:data,
                    '_declarations':lambda root:(report,a0,by_id),
                    'build_partitions':lambda *args:parts,
                    '_load_old_inner_evidence':lambda *args:evidence,
                    '_new_stage_probabilities':new_candidates,
                    '_build_bank':lambda *args:{}, '_select':lambda *args:selection,
                    '_load_inference':lambda *args:(test,sample), '_deploy_stage':deploy,
                }
                for name,value in overrides.items():
                    stack.enter_context(patch.object(gr,name,value))
                first = gr.run(config_path,root)
                self.assertEqual(first['status'],'PARTIAL')
                self.assertEqual(first['completed_grades'],['A0'])
                prior_csv = (root/'submissions/submission_grade_A0.csv').read_bytes()
                prior_selection = (root/'grades/A0/selection.joblib').read_bytes()
                failures[0] = False
                second = gr.run(config_path,root)
                self.assertEqual(second['status'],'COMPLETE')
                self.assertEqual(second['completed_grades'],['A0','A1'])
                self.assertEqual(deployments,['A0','A1'])
                self.assertEqual((root/'submissions/submission_grade_A0.csv').read_bytes(),prior_csv)
                self.assertEqual((root/'grades/A0/selection.joblib').read_bytes(),prior_selection)

    def test_frozen_selection_rejects_changed_membership(self):
        p = np.full((3,2),.5)
        c = MetaCandidate('R0-a',0,'A1',('C0',)*4,(1.,0.,0.,0.,0.),'arithmetic',p,.4,'x')
        selection = {'selected':c,'candidate_graph':{c.recipe_id:c},'manifests':[]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'selection.joblib'
            gr._freeze_selection(path,'A0',['C0'],selection)
            with self.assertRaisesRegex(ValueError,'membership'):
                gr._freeze_selection(path,'A0',['C0','C1'])


if __name__ == '__main__':
    unittest.main()
