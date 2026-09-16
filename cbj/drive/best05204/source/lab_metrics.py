"""Reviewed OOF/metric contracts reused without any training runner."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def objsha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    partial.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def save_npz(path, **arrays):
    path = Path(path)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    need(not any(a.dtype.hasobject for a in arrays.values()), "Object arrays are forbidden.")
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    partial.replace(path)

def prob(value):
    p = np.asarray(value, dtype=np.float64)
    need(p.ndim == 2 and len(p) > 0 and p.shape[1] > 1, "Invalid probability shape.")
    need(np.isfinite(p).all() and (p >= 0).all() and (p <= 1 + 1e-5).all(), "Invalid probability values.")
    need(np.allclose(p.sum(1), 1, atol=1e-5, rtol=0), "Invalid probability row sums.")
    return p / p.sum(1, keepdims=True)

def load_oof(path):
    with np.load(path, allow_pickle=False) as data:
        required = ("ids", "y", "groups", "folds", "class_order", "prob")
        need(set(required) <= set(data.files), "Missing OOF identity fields.")
        a = {k: data[k] for k in required}
    n, classes = len(a["ids"]), a["class_order"]
    need(all(a[k].ndim == 1 and len(a[k]) == n for k in ("ids", "y", "groups", "folds")), "OOF row shapes differ.")
    need(a["ids"].dtype.kind in "US" and a["groups"].dtype.kind in "US", "Text IDs/groups required.")
    need(n > 0 and len(set(a["ids"])) == n and np.all(a["ids"] != ""), "Duplicate/empty IDs.")
    need(classes.ndim == 1 and classes.dtype.kind in "US" and len(set(classes)) == len(classes), "Invalid class order.")
    need(a["y"].dtype.kind in "iu" and a["folds"].dtype.kind in "iu", "Integer y/folds required.")
    need(((a["y"] >= 0) & (a["y"] < len(classes))).all(), "Labels out of bounds.")
    assigned = {}
    for group, fold in zip(a["groups"], a["folds"]):
        need(group not in assigned or assigned[group] == fold, "Canonical group crosses folds.")
        assigned[group] = fold
    need(len(np.unique(a["folds"])) >= 2, "At least two stored folds required.")
    for fold in np.unique(a["folds"]):
        for mask in (a["folds"] == fold, a["folds"] != fold):
            need(set(a["y"][mask]) == set(range(len(classes))), "Stored fold lacks a class.")
    a["prob"] = prob(a["prob"])
    need(a["prob"].shape == (n, len(classes)), "OOF probability dimensions differ.")
    return a

def align(a, b):
    need(np.array_equal(a["class_order"], b["class_order"]), "Class order mismatch; no guessing.")
    need(set(a["ids"]) == set(b["ids"]), "OOF ID sets differ.")
    position = {v: i for i, v in enumerate(b["ids"])}
    take = np.array([position[v] for v in a["ids"]])
    b = {k: v if k == "class_order" else v[take] for k, v in b.items()}
    need(all(np.array_equal(a[k], b[k]) for k in ("ids", "y", "folds")), "OOF labels/folds mismatch.")
    forward, backward = {}, {}
    for u, v in zip(a["groups"], b["groups"]):
        need((u not in forward or forward[u] == v) and (v not in backward or backward[v] == u),
             "Canonical group partitions differ.")
        forward[u], backward[v] = v, u
    return b

def f1(y, predictions, k, weight=None):
    cm = np.bincount(y * k + predictions, weights=weight, minlength=k*k).reshape(k, k)
    denominator = cm.sum(0) + cm.sum(1)
    return cm, np.divide(2*cm.diagonal(), denominator, out=np.zeros(k, float), where=denominator > 0)

def metrics(a, p):
    y, classes = a["y"], a["class_order"]
    p = prob(p)
    cm, scores = f1(y, p.argmax(1), len(classes))
    true = p[np.arange(len(y)), y]
    confidence, correct = p.max(1), p.argmax(1) == y
    bins = np.minimum((confidence * 10).astype(int), 9)
    ece = sum(float((bins == i).mean() * abs(confidence[bins == i].mean()-correct[bins == i].mean()))
              for i in range(10) if (bins == i).any())
    per_class = []
    for i, cls in enumerate(classes):
        mask = y == i
        per_class.append(dict(SUBCLASS=str(cls), support_rows=int(mask.sum()),
            support_profiles=len(np.unique(a["groups"][mask])), predicted_rows=int(cm[:, i].sum()),
            f1=float(scores[i]), precision=float(cm[i, i]/cm[:, i].sum()) if cm[:, i].sum() else 0.,
            recall=float(cm[i, i]/cm[i].sum()) if cm[i].sum() else 0.,
            average_precision=float(average_precision_score(mask, p[:, i])) if mask.any() and (~mask).any() else None))
    fold_scores = []
    for fold in np.unique(a["folds"]):
        m = a["folds"] == fold
        fold_scores.append(float(f1(y[m], p[m].argmax(1), len(classes))[1].mean()))
    return dict(macro_f1=float(scores.mean()), fold_f1=fold_scores, fold_mean=float(np.mean(fold_scores)),
        fold_sd=float(np.std(fold_scores, ddof=1)), logloss=float(-np.log(np.clip(true, np.finfo(float).eps, 1)).mean()),
        brier_0_to_2=float((np.square(p).sum(1)-2*true+1).mean()), ece10=ece, per_class=per_class)

def correlation(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def changes(y, p, base):
    good, base_good = p.argmax(1) == y, base.argmax(1) == y
    per_pearson = [correlation(p[:, c], base[:, c]) for c in range(p.shape[1])]
    per_spearman = [correlation(rankdata(p[:, c]), rankdata(base[:, c])) for c in range(p.shape[1])]
    return dict(both_correct=int((good & base_good).sum()), rescued=int((good & ~base_good).sum()),
        damaged=int((~good & base_good).sum()), both_wrong=int((~good & ~base_good).sum()),
        disagreement=float((p.argmax(1) != base.argmax(1)).mean()),
        mean_total_variation=float((.5*np.abs(p-base).sum(1)).mean()),
        pearson_per_class=per_pearson, spearman_per_class=per_spearman,
        correlation_unit="Probability for the same class across patients; no encoded-label correlation.")

def bootstrap(a, p, base, count, seed):
    unique, inv = np.unique(a["groups"], return_inverse=True)
    rng, deltas = np.random.default_rng(seed), []
    pred, ref, k = p.argmax(1), base.argmax(1), p.shape[1]
    for _ in range(count):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[inv]
        deltas.append(float(f1(a["y"], pred, k, weights)[1].mean() - f1(a["y"], ref, k, weights)[1].mean()))
    return dict(unit="canonical_group", resamples=count, ci95=np.quantile(deltas, [.025, .975]).tolist(),
        scope="Conditional on fixed development predictions; excludes retraining/selection uncertainty.")
