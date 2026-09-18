"""TRAIN-only paired screen: prefix every mutation event with its gene.

Only NLP document representation changes. Reuse aligned exp56 XGB OOF.
Vectorizer settings, SVC, calibration and blend match the preserved baseline.
Outer validation is never used for fitting. Existing inner calibration protocol
uses an outer-train-fitted vectorizer in both arms, not inner-fold refitting.
This is a controlled outer-CV screen, not a new calibration-method comparison.
No evaluation input access; no parameter search. Exclusive output directories.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import platform
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def align(df, train):
    assert not df.row_index.duplicated().any()
    df = df.set_index('row_index').loc[np.arange(len(train))].reset_index()
    assert np.array_equal(df.ID, train.ID)
    assert np.array_equal(df.true_subclass, train.SUBCLASS)
    return df

def train_predict(docs, tr, va, y, seed):
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=2,
        max_features=120000, sublinear_tf=True, norm='l2', lowercase=False,
        dtype=np.float32)
    xtr = vec.fit_transform(docs[tr])
    xva = vec.transform(docs[va])
    model = CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced',
        random_state=seed, max_iter=5000), method='sigmoid', cv=3, n_jobs=1)
    model.fit(xtr, y[tr])
    prob = model.predict_proba(xva)
    assert np.isfinite(prob).all() and np.allclose(prob.sum(axis=1),1)
    return prob

def run(seed):
    start = time.monotonic()
    out = ROOT / f'exp_63_nlp_event_gene_seed{seed}'
    out.mkdir(exist_ok=False)
    train_path = ROOT / 'train.csv'
    prob_path = ROOT / f'exp_56_seed{seed}_class_weight_oof_probabilities.csv'
    labels_path = ROOT / f'exp_56_seed{seed}_class_weight_predictions.csv'
    manifest = dict(seed=seed, folds=5, gate_delta=.005, gate_folds=4,
        change='NLP per-event gene prefix only', blend=[.8,.2],
        reuse='weighted XGB exp56 same patient/fold OOF',
        calibration='preserved cv=3; vectorizer fitted on outer training only',
        evaluation_inputs_used=False, python=platform.python_version(),
        sklearn=sklearn.__version__, source_sha256=digest(Path(__file__)),
        inputs={p.name:digest(p) for p in [train_path,prob_path,labels_path]})
    with (out/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2)
    train = pd.read_csv(train_path).reset_index(drop=True)
    enc = LabelEncoder().fit(train.SUBCLASS)
    classes = enc.classes_
    y = enc.transform(train.SUBCLASS)
    saved = align(pd.read_csv(prob_path),train)
    labels = align(pd.read_csv(labels_path),train)
    base = saved[[f'candidate_ensemble_prob_{c}' for c in classes]].to_numpy()
    xgb = saved[[f'candidate_xgb_prob_{c}' for c in classes]].to_numpy()
    old_nlp = saved[[f'nlp_prob_{c}' for c in classes]].to_numpy()
    assert np.allclose(base,.8*xgb+.2*old_nlp,atol=2e-7)
    assert np.array_equal(classes[base.argmax(1)], labels.candidate_ensemble_pred)
    for a in (base,xgb,old_nlp):
        assert np.isfinite(a).all() and (a>=0).all()
        assert np.allclose(a.sum(1),1,atol=2e-5)
    genes = [c for c in train if c not in ('ID','SUBCLASS')]
    old_tokens = [[] for _ in y]
    new_tokens = [[] for _ in y]
    multi_cells = extra = 0
    cells = train[genes].stack().dropna()
    cells = cells[cells.ne('WT')].astype(str)
    for (row,gene),value in cells.items():
        old_tokens[row].append(f'{gene}_{value}')
        events = value.split()
        new_tokens[row].extend(f'{gene}_{event}' for event in events)
        multi_cells += len(events)>1
        extra += max(len(events)-1,0)
    docs = np.array([' '.join(t) for t in new_tokens],dtype=object)
    old_docs = np.array([' '.join(t) for t in old_tokens],dtype=object)
    print(f'exp63 seed={seed}: changed patients={sum(docs!=old_docs)}, '
          f'multi cells={multi_cells}, restored gene prefixes={extra}',flush=True)
    new = np.zeros_like(base)
    rows=[]
    folds = np.zeros(len(y),dtype=int)
    cv = StratifiedKFold(5,shuffle=True,random_state=seed)
    for fold,(tr,va) in enumerate(cv.split(np.zeros(len(y)),y),1):
        tick=time.monotonic()
        assert np.array_equal(np.flatnonzero(labels.fold.to_numpy()==fold),va)
        if fold==1:
            audit = train_predict(old_docs,tr,va,y,seed)
            err=float(np.max(np.abs(audit-old_nlp[va])))
            assert np.allclose(audit,old_nlp[va],atol=2e-6), err
            print(f'Original NLP reproduction PASS max abs diff={err:.3g}',flush=True)
        new[va] = train_predict(docs,tr,va,y,seed)
        folds[va]=fold
        blend=.8*xgb[va]+.2*new[va]
        score=lambda prob: float(f1_score(y[va],prob.argmax(1),average='macro',zero_division=0))
        row=dict(seed=seed,fold=fold,baseline_f1=score(base[va]),candidate_f1=score(blend),
                 baseline_nlp_f1=score(old_nlp[va]),candidate_nlp_f1=score(new[va]),
                 seconds=time.monotonic()-tick)
        row['delta']=row['candidate_f1']-row['baseline_f1']
        rows.append(row)
        checkpoint=pd.DataFrame(new[va],columns=[f'nlp_prob_{c}' for c in classes])
        checkpoint.insert(0,'row_index',va)
        checkpoint.to_csv(out/f'fold_{fold}_probabilities.csv',index=False,mode='x')
        print(json.dumps(row),flush=True)
    blend=.8*xgb+.2*new
    bp,cp=base.argmax(1),blend.argmax(1)
    score=lambda pred: float(f1_score(y,pred,average='macro',zero_division=0))
    summary=dict(seed=seed,baseline_f1=score(bp),candidate_f1=score(cp),
        baseline_nlp_f1=score(old_nlp.argmax(1)),candidate_nlp_f1=score(new.argmax(1)),
        improved_folds=sum(r['delta']>0 for r in rows),
        rescue=int(((bp!=y)&(cp==y)).sum()),damage=int(((bp==y)&(cp!=y)).sum()),
        seconds=time.monotonic()-start)
    summary['delta']=summary['candidate_f1']-summary['baseline_f1']
    summary['pass']=summary['delta']>=.005 and summary['improved_folds']>=4
    with (out/'summary.json').open('x') as f:
        json.dump(summary,f,indent=2)
    pd.DataFrame(rows).to_csv(out/'fold_results.csv',index=False,mode='x')
    b=precision_recall_fscore_support(y,bp,labels=np.arange(len(classes)),zero_division=0)
    c=precision_recall_fscore_support(y,cp,labels=np.arange(len(classes)),zero_division=0)
    pd.DataFrame(dict(subclass=classes,support=b[3],baseline_f1=b[2],candidate_f1=c[2],
                     delta=c[2]-b[2])).to_csv(out/'classwise.csv',index=False,mode='x')
    meta=pd.DataFrame(dict(row_index=np.arange(len(y)),ID=train.ID,true_subclass=train.SUBCLASS,
        fold=folds,seed=seed,baseline_pred=classes[bp],candidate_pred=classes[cp]))
    pd.concat([meta,pd.DataFrame(base,columns=[f'baseline_prob_{c}' for c in classes]),
        pd.DataFrame(blend,columns=[f'candidate_prob_{c}' for c in classes]),
        pd.DataFrame(new,columns=[f'nlp_prob_{c}' for c in classes])],axis=1).to_csv(
            out/'oof_probabilities.csv',index=False,mode='x')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=19,choices=[19,193,2029])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.seed)
