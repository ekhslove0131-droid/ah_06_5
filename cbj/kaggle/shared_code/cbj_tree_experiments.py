"""Config-driven tree experiments. Imports are centralized; no competition API calls."""
# All library imports belong here, never inside functions or notebook result cells.
import argparse
import gzip
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import classification_report, f1_score, log_loss
from cbj_composition_features import composition_features

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

MODEL_CLASSES = {'xgboost': XGBClassifier, 'lightgbm': LGBMClassifier, 'catboost': CatBoostClassifier}


def read_json(path):
    with (gzip.open(path, 'rt') if str(path).endswith('.gz') else open(path)) as handle:
        return json.load(handle)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_input(base, filename, expected_sha=None):
    """Resolve exactly one artifact; ambiguous matches fail instead of taking the first."""
    candidates = [p for p in Path(base).rglob(filename) if not expected_sha or digest(p) == expected_sha]
    if len(candidates) != 1:
        raise ValueError(f'Expected exactly one {filename}, found {len(candidates)}')
    return candidates[0]


def feature_matrix(matrix, names, feature_config, stage_genes, memory_limit_mb):
    """Same explicit numerical zeros for all backends. Missing flags stay separate."""
    indices = [i for i, name in enumerate(names) if any(name.startswith(p) for p in feature_config['prefixes'])]
    if not indices:
        raise ValueError('No configured feature columns')
    index = {name: i for i, name in enumerate(names)}
    derived, derived_names = [], []

    def column_sum(keys):
        cols = [index[k] for k in keys if k in index]
        return np.asarray(matrix[:, cols].sum(axis=1)).ravel() if cols else np.zeros(matrix.shape[0])

    def add(name, values):
        derived_names.append('derived|' + name)
        derived.append(np.asarray(values, dtype=np.float32))

    gene_keys = [name for name in names if name.startswith('gene|') and name.endswith('|nonWT')]
    burden = column_sum(gene_keys)
    if feature_config['derived_counts']:
        add('mutated_gene_count', burden)
        add('mutated_gene_log1p', np.log1p(burden))
        add('no_observed_mutation', burden == 0)
        types = sorted({name.split('|')[-1] for name in names if name.startswith('type|')})
        for kind in types:
            add('gene_count_type_' + kind, column_sum([n for n in names if n.startswith('type|') and n.endswith('|' + kind)]))
        mapped = set().union(*(set(gs) for gs in stage_genes.values()))
        add('mapped_unique_gene_count', column_sum(['gene|' + g + '|nonWT' for g in sorted(mapped)]))
        for stage, genes in stage_genes.items():
            count = column_sum(['gene|' + g + '|nonWT' for g in genes])
            add(stage + '_gene_count', count)
            add(stage + '_share_of_burden', np.divide(count, burden, out=np.zeros_like(count), where=burden > 0))
            add(stage + '_outside_gene_count', burden - count)
    required_mb = matrix.shape[0] * (len(indices) + len(derived)) * np.dtype(np.float32).itemsize / 2**20
    if required_mb > memory_limit_mb:
        raise MemoryError(f'Dense feature matrix {required_mb:.1f} MiB exceeds configured limit {memory_limit_mb}')
    out = np.empty((matrix.shape[0], len(indices) + len(derived)), dtype=np.float32)
    out[:, :len(indices)] = matrix[:, indices].toarray()
    for j, values in enumerate(derived):
        out[:, len(indices) + j] = values
    if not np.isfinite(out).all():
        raise ValueError('Nonfinite engineered features')
    output_names = [names[i] for i in indices] + derived_names
    if feature_config.get("composition_summaries", False):
        extra, extra_names = composition_features(out, output_names)
        out = np.column_stack([out, extra])
        output_names += extra_names
    return out, output_names


def new_model(spec, n_classes, seed, threads):
    family = spec['family']
    cls = MODEL_CLASSES[family]
    if cls is None:
        raise RuntimeError(f'{family} is not installed; use the configured Kaggle runtime')
    parameters = dict(spec['params'])
    if family == 'catboost':
        parameters.update(random_seed=seed, thread_count=threads, classes_count=n_classes)
    else:
        parameters.update(random_state=seed, n_jobs=threads, num_class=n_classes)
    return cls(**parameters)


def metrics(y, probabilities, classes):
    p = np.asarray(probabilities, dtype=float)
    if p.shape != (len(y), len(classes)) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid class probability matrix')
    p = p / p.sum(axis=1, keepdims=True)
    prediction = p.argmax(axis=1)
    return {'macro_f1': float(f1_score(y, prediction, labels=np.arange(len(classes)), average='macro', zero_division=0)),
            'log_loss': float(log_loss(y, p, labels=np.arange(len(classes))))}


def anchor_distance(candidate, reference, y, classes):
    """Diagnostics only: no distance metric enters submission eligibility."""
    a, b = np.asarray(candidate), np.asarray(reference)
    ar, br = a.argmax(1), b.argmax(1)
    rows = []
    for i, label in enumerate(classes):
        variable = np.ptp(a[:, i]) > 0 and np.ptp(b[:, i]) > 0
        rows.append({'SUBCLASS': label, 'pearson': float(pearsonr(a[:, i], b[:, i]).statistic) if variable else None,
                     'spearman': float(spearmanr(a[:, i], b[:, i]).statistic) if variable else None})
    return {'label_disagreement': float(np.mean(ar != br)), 'mean_total_variation': float(np.abs(a - b).sum(1).mean() / 2),
            'both_correct': int(((ar == y) & (br == y)).sum()),
            'candidate_only_correct': int(((ar == y) & (br != y)).sum()),
            'anchor_only_correct': int(((ar != y) & (br == y)).sum()),
            'both_wrong': int(((ar != y) & (br != y)).sum()), 'per_class_correlations': rows}


def load_context(config, input_root):
    sources = config['sources']
    preflight = resolve_input(input_root, sources['preflight_name'], sources['preflight_sha256'])
    root = preflight.parent
    prepared = read_json(root / sources['prepared_config'])
    for filename, checksum in read_json(preflight)['file_sha256'].items():
        if digest(root / filename) != checksum:
            raise ValueError('Changed preparation artifact: ' + filename)
    assignments = pd.read_csv(root / sources['fold_assignments'], dtype={'ID': str})
    classes = prepared['class_order']
    if not assignments.ID.is_unique or assignments.SUBCLASS.isna().any():
        raise ValueError('Invalid OOF identities or labels')
    if not assignments.groupby('canonical_hash').fold.nunique().eq(1).all():
        raise ValueError('Canonical profiles cross folds')
    y = assignments.SUBCLASS.map({c: i for i, c in enumerate(classes)})
    if y.isna().any():
        raise ValueError('Unknown label')
    return root, prepared, assignments, y.to_numpy(dtype=int)


def validate(config, input_root, out):
    root, prepared, assignments, y = load_context(config, input_root)
    fold = sorted(assignments.fold.unique())[0]
    enc = read_json(root / config['files']['encoder'].format(fold=fold))
    source = load_npz(root / config['files']['valid_matrix'].format(fold=fold))
    x, names = feature_matrix(source[:config['validation']['rows']], enc['names'], config['features'], prepared['stage_genes'], config['runtime']['matrix_limit_mb'])
    info = {'mode': 'validation_only', 'training_performed': False, 'rows': len(assignments), 'classes': len(prepared['class_order']),
            'folds': len(assignments.fold.unique()), 'feature_count': len(names), 'preview_shape': list(x.shape),
            'families_available': {k: v is not None for k, v in MODEL_CLASSES.items()}, 'config': config}
    if config['validation'].get('backend_smoke'):
        v = config['validation']
        rng = np.random.default_rng(config['runtime']['seed'])
        toy_x = rng.normal(size=(v['synthetic_rows'], v['synthetic_features'])).astype(np.float32)
        toy_y = np.arange(v['synthetic_rows']) % v['synthetic_classes']
        info['synthetic_backend_checks'] = {}
        for original in config['experiments']:
            spec = {'family': original['family'], 'params': dict(original['params'])}
            round_key = 'iterations' if spec['family'] == 'catboost' else 'n_estimators'
            spec['params'][round_key] = v['synthetic_rounds']
            model = new_model(spec, v['synthetic_classes'], config['runtime']['seed'], config['runtime']['threads'])
            model.fit(toy_x, toy_y)
            probabilities = model.predict_proba(toy_x[:v['rows']])
            metrics(toy_y[:v['rows']], probabilities, list(range(v['synthetic_classes'])))
            info['synthetic_backend_checks'][spec['family']] = 'PASS (synthetic only, no real model score)'
        info['training_performed'] = 'synthetic backend checks only; no real data fitting'
    (out / 'validation.json').write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != 'config'}, indent=2))


def evaluate(config, input_root, out):
    root, prepared, assignments, y = load_context(config, input_root)
    classes, folds = prepared['class_order'], sorted(assignments.fold.unique())
    reference = None
    reference_name = config['sources'].get('reference_oof')
    if reference_name:
        ref = pd.read_csv(resolve_input(input_root, reference_name), dtype={'ID': str})
        if not ref.ID.is_unique or set(ref.ID) != set(assignments.ID):
            raise ValueError('Anchor reference ID mismatch')
        ref = ref.set_index('ID').loc[assignments.ID]
        if ref.SUBCLASS.tolist() != assignments.SUBCLASS.tolist() or ref.fold.tolist() != assignments.fold.tolist():
            raise ValueError('Anchor reference labels/folds mismatch')
        reference = ref[['p_' + c for c in classes]].to_numpy()
        metrics(y, reference, classes)
    results = []
    for spec in config['experiments']:
        folder = out / spec['name']
        folder.mkdir()
        p = np.full((len(y), len(classes)), np.nan)
        fold_results = []
        for fold in folds:
            tr, va = np.flatnonzero(assignments.fold != fold), np.flatnonzero(assignments.fold == fold)
            enc = read_json(root / config['files']['encoder'].format(fold=fold))
            if set(enc['fit_ids']) != set(assignments.ID.iloc[tr]):
                raise ValueError('Encoder fit identities mismatch')
            xt, names = feature_matrix(load_npz(root / config['files']['train_matrix'].format(fold=fold)), enc['names'], config['features'], prepared['stage_genes'], config['runtime']['matrix_limit_mb'])
            xv, valid_names = feature_matrix(load_npz(root / config['files']['valid_matrix'].format(fold=fold)), enc['names'], config['features'], prepared['stage_genes'], config['runtime']['matrix_limit_mb'])
            if names != valid_names or len(xt) != len(tr) or len(xv) != len(va):
                raise ValueError('Matrix alignment mismatch')
            model = new_model(spec, len(classes), config['runtime']['seed'], config['runtime']['threads'])
            start = perf_counter()
            # Fixed rounds: outer validation labels are not used for early stopping.
            model.fit(xt, y[tr])
            if spec['family'] == 'catboost':
                (folder / f'fold_{fold}_model_params.json').write_text(json.dumps(model.get_all_params(), indent=2))
            if list(model.classes_) != list(range(len(classes))):
                raise ValueError('Model class order mismatch')
            p[va] = model.predict_proba(xv)
            result = {'fold': int(fold), 'n': len(va), 'seconds': perf_counter() - start, **metrics(y[va], p[va], classes)}
            fold_results.append(result)
            joblib.dump(model, folder / f'fold_{fold}.joblib', compress=3)
            print(spec['name'], json.dumps(result), flush=True)
            del xt, xv, model
        pooled = metrics(y, p, classes)
        result = {'name': spec['name'], 'family': spec['family'], **pooled, 'folds': fold_results,
                  'fold_std': float(np.std([r['macro_f1'] for r in fold_results], ddof=1))}
        if reference is not None:
            (folder / 'anchor_distance.json').write_text(json.dumps(anchor_distance(p, reference, y, classes), indent=2))
        table = assignments.copy()
        for i, label in enumerate(classes):
            table['p_' + label] = p[:, i]
        table.to_csv(folder / 'oof.csv', index=False)
        pd.DataFrame(classification_report(y, p.argmax(1), labels=np.arange(len(classes)), target_names=classes, output_dict=True, zero_division=0)).T.to_csv(folder / 'per_class.csv')
        (folder / 'metrics.json').write_text(json.dumps(result, indent=2))
        results.append(result)
    ranked = sorted(results, key=lambda r: r['macro_f1'], reverse=True)
    policy = config['selection']
    winner = ranked[0]
    gates = {'positive_macro_f1': winner['macro_f1'] > 0}
    if policy.get('internal_floor') is not None:
        gates['above_internal_floor'] = winner['macro_f1'] >= policy['internal_floor']
    if policy.get('max_fold_std') is not None:
        gates['fold_stability'] = winner['fold_std'] <= policy['max_fold_std']
    incumbent = policy.get('incumbent_macro_f1')
    if incumbent is not None:
        gates['improves_current_best'] = winner['macro_f1'] >= incumbent + policy['minimum_improvement']
    decision = {'ranked_results': ranked, 'winner': winner['name'], 'gates': gates,
                'near_ties': [r['name'] for r in ranked[1:] if winner['macro_f1'] - r['macro_f1'] < policy['near_tie_band']],
                'submission_candidate': all(gates.values()), 'competition_submitted': False,
                'anchor_distance_is_gate': False, 'test_score_used_for_selection': False,
                'selection_policy': policy}
    (out / 'selection.json').write_text(json.dumps(decision, indent=2))
    return decision


def make_submission(config, input_root, out, decision):
    if not decision['submission_candidate']:
        raise ValueError('Internal selection gates did not pass; no submission CSV generated')
    root, prepared, assignments, y = load_context(config, input_root)
    spec = next(s for s in config['experiments'] if s['name'] == decision['winner'])
    enc = read_json(root / config['files']['full_encoder'])
    if set(enc['fit_ids']) != set(assignments.ID):
        raise ValueError('Full encoder fit identities mismatch')
    x, names = feature_matrix(load_npz(root / config['files']['full_matrix']), enc['names'], config['features'], prepared['stage_genes'], config['runtime']['matrix_limit_mb'])
    model = new_model(spec, len(prepared['class_order']), config['runtime']['seed'], config['runtime']['threads'])
    model.fit(x, y)
    if list(model.classes_) != list(range(len(prepared['class_order']))):
        raise ValueError('Full model class order mismatch')
    del x
    test, test_names = feature_matrix(load_npz(root / config['files']['test_matrix']), enc['names'], config['features'], prepared['stage_genes'], config['runtime']['matrix_limit_mb'])
    if test_names != names:
        raise ValueError('Test feature schema mismatch')
    sample = pd.read_csv(resolve_input(input_root, config['sources']['sample_submission']), dtype=str, keep_default_na=False)
    ids = pd.read_csv(root / config['files']['test_ids'], dtype={'ID': str}).ID
    if list(sample.columns) != config['submission']['columns'] or not sample.ID.is_unique or not ids.is_unique or set(ids) != set(sample.ID) or len(test) != len(ids):
        raise ValueError('Submission schema or ID mismatch')
    probabilities = model.predict_proba(test)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
        raise ValueError('Invalid test probabilities')
    labels = pd.Series(np.asarray(prepared['class_order'])[probabilities.argmax(1)], index=ids)
    sample['SUBCLASS'] = sample.ID.map(labels)
    if sample.SUBCLASS.isna().any():
        raise ValueError('Missing submission label')
    path = out / config['submission']['filename'].format(experiment=spec['name'])
    sample.to_csv(path, index=False)
    pd.testing.assert_frame_equal(pd.read_csv(path, dtype=str, keep_default_na=False), sample)
    joblib.dump(model, out / 'full_model.joblib', compress=3)
    (out / 'submission_receipt.json').write_text(json.dumps({'rows': len(sample), 'sha256': digest(path), 'source_model': spec['name'],
        'class_order': prepared['class_order'], 'feature_names': names, 'competition_submitted': False}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['validate', 'evaluate'], default='validate')
    parser.add_argument('--make-submission', action='store_true')
    args = parser.parse_args()
    if args.make_submission and args.mode != 'evaluate':
        raise ValueError('Submission requires successful evaluation')
    if args.mode == 'evaluate' and not Path('/kaggle/input').exists():
        raise RuntimeError('Real-data evaluation is configured for Kaggle')
    config = read_json(args.config)
    gpu_inventory = None
    if config['runtime'].get('expected_gpu_count'):
        probe = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        gpu_inventory = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        if len(gpu_inventory) != config['runtime']['expected_gpu_count'] or not all(config['runtime']['expected_gpu_name'] in line for line in gpu_inventory):
            raise RuntimeError('Requested GPU allocation unavailable: ' + str(gpu_inventory))
        print('GPU allocation:', gpu_inventory, flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    if gpu_inventory is not None:
        (args.output / 'gpu_inventory.json').write_text(json.dumps(gpu_inventory, indent=2))
    (args.output / 'config.json').write_text(json.dumps(config, indent=2))
    (args.output / 'provenance.json').write_text(json.dumps({'config_sha256': digest(args.config), 'code_sha256': digest(__file__),
         'python': platform.python_version(), 'packages': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'scikit-learn'] + [name for name, cls in MODEL_CLASSES.items() if cls is not None]}}, indent=2))
    if args.mode == 'validate':
        validate(config, args.input_root, args.output)
    else:
        decision = evaluate(config, args.input_root, args.output)
        if args.make_submission and decision['submission_candidate']:
            make_submission(config, args.input_root, args.output, decision)


if __name__ == '__main__':
    main()
