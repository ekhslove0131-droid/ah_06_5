"""TRAIN-only controlled capacity screen: max_depth 3 versus 6.

Same multi-event representation, cross-fitted signature, class weights and NLP.
No new position bins or NLP changes from exp63/64. No parameter search.
Every baseline fold reproduces saved exp56 and records resubstitution metrics.
Train scores are diagnostic only, never the selection criterion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import re
import time
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits
import xgboost
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
HASH_SIZE = 2**15

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def align(df, train):
    assert not df.row_index.duplicated().any()
    df = df.set_index('row_index').loc[np.arange(len(train))].reset_index()
    assert np.array_equal(df.ID,train.ID)
    assert np.array_equal(df.true_subclass,train.SUBCLASS)
    return df

def mutation_type(token):
    if 'fs' in token.lower():
        return 2
    if '*' in token:
        return 3
    aa = re.search(r'([A-Z])(\d+)([A-Z])', token)
    if aa:
        before, _, after = aa.groups()
        return 1 if before==after else 0
    return 4

def centroid(X,y,rows,nclasses):
    c = np.zeros((nclasses,X.shape[1]),dtype=np.float32)
    for label in range(nclasses):
        selected=rows[y[rows]==label]
        if len(selected):
            c[label]=np.asarray(X[selected].mean(axis=0)).ravel().astype(np.float32)
    return c

def signature(X,c):
    dot=np.asarray(X@c.T,dtype=np.float32)
    row_norm=np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel()).astype(np.float32)
    c_norm=np.linalg.norm(c,axis=1).astype(np.float32)
    denom=row_norm[:,None]*c_norm[None,:]
    return np.divide(dot,denom,out=np.zeros_like(dot),where=denom>0).astype(np.float32)

def crossfit_signature(X,y,tr,va,seed,nclasses):
    train_sig=np.zeros((len(tr),nclasses),dtype=np.float32)
    inner=StratifiedKFold(5,shuffle=True,random_state=seed)
    for it,iv in inner.split(np.zeros(len(tr)),y[tr]):
        train_sig[iv]=signature(X[tr[iv]],centroid(X,y,tr[it],nclasses))
    return train_sig,signature(X[va],centroid(X,y,tr,nclasses))

def model(seed, depth):
    return XGBClassifier(n_estimators=200,learning_rate=.10,max_depth=depth,
        random_state=seed,n_jobs=2,tree_method='hist')

def run(seed):
    start=time.monotonic()
    out=ROOT/f'exp_65_depth6_seed{seed}'
    out.mkdir(exist_ok=False)
    train_path=ROOT/'train.csv'
    prob_path=ROOT/f'exp_56_seed{seed}_class_weight_oof_probabilities.csv'
    labels_path=ROOT/f'exp_56_seed{seed}_class_weight_predictions.csv'
    manifest=dict(seed=seed,folds=5,baseline_depth=3,candidate_depth=6,hash_size=HASH_SIZE,
        change='max_depth 3 to 6 only',
        gate_delta=.005,gate_improved_folds=4,blend=[.8,.2],
        evaluation_input_access=False,external_data=False,parameter_search=False,
        python=platform.python_version(),sklearn=sklearn.__version__,xgboost=xgboost.__version__,
        script_sha256=digest(Path(__file__)),
        inputs={p.name:digest(p) for p in (train_path,prob_path,labels_path)})
    with (out/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2)
    train=pd.read_csv(train_path).reset_index(drop=True)
    enc=LabelEncoder().fit(train.SUBCLASS)
    classes=enc.classes_
    y=enc.transform(train.SUBCLASS)
    saved=align(pd.read_csv(prob_path),train)
    old_labels=align(pd.read_csv(labels_path),train)
    old_xgb=saved[[f'candidate_xgb_prob_{c}' for c in classes]].to_numpy()
    old_nlp=saved[[f'nlp_prob_{c}' for c in classes]].to_numpy()
    baseline=saved[[f'candidate_ensemble_prob_{c}' for c in classes]].to_numpy()
    assert np.allclose(baseline,.8*old_xgb+.2*old_nlp,atol=2e-7)
    assert np.array_equal(classes[baseline.argmax(1)],old_labels.candidate_ensemble_pred)
    for a in (old_xgb,old_nlp,baseline):
        assert np.isfinite(a).all() and (a>=0).all()
        assert np.allclose(a.sum(1),1,atol=2e-5)
    genes=[c for c in train if c not in ('ID','SUBCLASS')]
    binary_np=(train[genes].notna()&train[genes].ne('WT')).to_numpy(dtype=np.uint8)
    X=sparse.csr_matrix(binary_np,dtype=np.float32)
    count=binary_np.sum(axis=1).astype(np.float32).reshape(-1,1)
    types=np.zeros((len(y),5),dtype=np.float32)
    exact=[[] for _ in y]
    cells=train[genes].stack().dropna()
    cells=cells[cells.ne('WT')].astype(str)
    for (row,gene),value in cells.items():
        for token in value.split():
            types[row,mutation_type(token)]+=1
            match=re.search(r'(\d+)',token)
            if match:
                pos=int(match.group(1))
                exact[row].append(f'{gene}_POS{pos}')
    hasher=FeatureHasher(n_features=HASH_SIZE,input_type='string',alternate_sign=False)
    base=sparse.hstack([X,sparse.csr_matrix(count),sparse.csr_matrix(types),
                       hasher.transform(exact)],format='csr')
    print(f'exp65 seed={seed} | depth3 vs depth6 | TRAIN only | base={base.shape} | same NLP',flush=True)
    preds=np.zeros_like(baseline)
    fold_index=np.zeros(len(y),dtype=int)
    results=[]
    outer=StratifiedKFold(5,shuffle=True,random_state=seed)
    for fold,(tr,va) in enumerate(outer.split(np.zeros(len(y)),y),1):
        tick=time.monotonic()
        assert np.array_equal(np.flatnonzero(old_labels.fold.to_numpy()==fold),va)
        st,sv=crossfit_signature(X,y,tr,va,seed,len(classes))
        bt=sparse.hstack([base[tr],sparse.csr_matrix(st)],format='csr')
        bv=sparse.hstack([base[va],sparse.csr_matrix(sv)],format='csr')
        counts=np.bincount(y[tr],minlength=len(classes)).astype(np.float64)
        assert (counts>0).all()
        weights=1/np.sqrt(counts[y[tr]])
        weights/=weights.mean()
        print(f'Seed {seed} fold {fold}: fitting baseline depth3',flush=True)
        audit=model(seed,3)
        audit.fit(bt,y[tr],sample_weight=weights)
        audit_prob=audit.predict_proba(bv)
        err=float(np.max(np.abs(audit_prob-old_xgb[va])))
        exact_labels=bool(np.array_equal(audit_prob.argmax(1),old_xgb[va].argmax(1)))
        assert np.allclose(audit_prob,old_xgb[va],atol=2e-6) and exact_labels, err
        baseline_train_f1=float(f1_score(y[tr],audit.predict(bt),average='macro',zero_division=0))
        with (out/f'baseline_reproduction_fold{fold}.json').open('x') as f:
            json.dump(dict(fold=fold,max_probability_difference=err,exact_labels=exact_labels,
                baseline_xgb_train_f1=baseline_train_f1),f,indent=2)
        print(f'Baseline reproduction PASS; training F1={baseline_train_f1:.6f}; fitting depth6',flush=True)
        candidate=model(seed,6)
        candidate.fit(bt,y[tr],sample_weight=weights)
        candidate_train_f1=float(f1_score(y[tr],candidate.predict(bt),average='macro',zero_division=0))
        prob=candidate.predict_proba(bv)
        assert np.isfinite(prob).all() and np.allclose(prob.sum(1),1,atol=2e-5)
        preds[va]=prob
        fold_index[va]=fold
        combined=.8*prob+.2*old_nlp[va]
        score=lambda p:float(f1_score(y[va],p.argmax(1),average='macro',zero_division=0))
        row=dict(seed=seed,fold=fold,baseline_f1=score(baseline[va]),candidate_f1=score(combined),
                 baseline_xgb_f1=score(old_xgb[va]),candidate_xgb_f1=score(prob),
                 baseline_xgb_train_f1=baseline_train_f1,candidate_xgb_train_f1=candidate_train_f1,
                 seconds=time.monotonic()-tick)
        row['delta']=row['candidate_f1']-row['baseline_f1']
        results.append(row)
        checkpoint=pd.DataFrame(prob,columns=[f'xgb_prob_{c}' for c in classes])
        checkpoint.insert(0,'row_index',va)
        checkpoint.to_csv(out/f'fold_{fold}_probabilities.csv',index=False,mode='x')
        print(json.dumps(row),flush=True)
    combined=.8*preds+.2*old_nlp
    bp,cp=baseline.argmax(1),combined.argmax(1)
    score=lambda p:float(f1_score(y,p,average='macro',zero_division=0))
    summary=dict(seed=seed,baseline_f1=score(bp),candidate_f1=score(cp),
        baseline_xgb_f1=score(old_xgb.argmax(1)),candidate_xgb_f1=score(preds.argmax(1)),
        baseline_mean_train_f1=float(np.mean([r['baseline_xgb_train_f1'] for r in results])),
        candidate_mean_train_f1=float(np.mean([r['candidate_xgb_train_f1'] for r in results])),
        improved_folds=sum(r['delta']>0 for r in results),
        rescue=int(((bp!=y)&(cp==y)).sum()),damage=int(((bp==y)&(cp!=y)).sum()),
        seconds=time.monotonic()-start)
    summary['delta']=summary['candidate_f1']-summary['baseline_f1']
    summary['pass']=summary['delta']>=.005 and summary['improved_folds']>=4
    with (out/'summary.json').open('x') as f:
        json.dump(summary,f,indent=2)
    pd.DataFrame(results).to_csv(out/'fold_results.csv',index=False,mode='x')
    b=precision_recall_fscore_support(y,bp,labels=np.arange(len(classes)),zero_division=0)
    c=precision_recall_fscore_support(y,cp,labels=np.arange(len(classes)),zero_division=0)
    pd.DataFrame(dict(subclass=classes,support=b[3],baseline_f1=b[2],candidate_f1=c[2],
        delta=c[2]-b[2])).to_csv(out/'classwise.csv',index=False,mode='x')
    meta=pd.DataFrame(dict(row_index=np.arange(len(y)),ID=train.ID,true_subclass=train.SUBCLASS,
        seed=seed,fold=fold_index,baseline_pred=classes[bp],candidate_pred=classes[cp]))
    pd.concat([meta,pd.DataFrame(baseline,columns=[f'baseline_prob_{c}' for c in classes]),
        pd.DataFrame(combined,columns=[f'candidate_prob_{c}' for c in classes]),
        pd.DataFrame(preds,columns=[f'xgb_prob_{c}' for c in classes])],axis=1).to_csv(
            out/'oof_probabilities.csv',index=False,mode='x')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=19,choices=[19,193,2029])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.seed)

