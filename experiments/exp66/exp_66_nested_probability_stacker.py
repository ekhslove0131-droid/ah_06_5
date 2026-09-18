"""Our TRAIN-only nested stacking screen, independently implemented.

Outer validation is excluded from every meta-training feature generator.
Saved exp56 OOF is used ONLY for the corresponding outer validation rows,
never as meta-training data. Inner 3-fold probabilities are freshly fitted.
Only combiner changes: fixed 80:20 versus standardized probability logistic
regression (52 inputs, C=1, balanced weights). No gating or class quotas.
The underlying NLP/calibration protocol is preserved; vectorizer fits only
the base learner's allowed training partition, as in the baseline.
"""
from pathlib import Path
import argparse
import json
import platform
import re
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from threadpoolctl import threadpool_limits

# These project helpers have main guards; importing never starts an experiment.
# Reuse only preserved depth3 / original-document helpers, not failed candidates.
from exp_65_model_capacity_screen import mutation_type, crossfit_signature, model, digest
from exp_63_nlp_event_gene_screen import train_predict as original_nlp_predict

ROOT=Path(__file__).resolve().parent

def align(frame,train):
    assert not frame.row_index.duplicated().any()
    frame=frame.set_index('row_index').loc[np.arange(len(train))].reset_index().copy()
    assert np.array_equal(frame.ID,train.ID)
    assert np.array_equal(frame.true_subclass,train.SUBCLASS)
    return frame

def features(train):
    genes=[c for c in train if c not in ('ID','SUBCLASS')]
    binary_np=(train[genes].notna()&train[genes].ne('WT')).to_numpy(dtype=np.uint8)
    binary=sparse.csr_matrix(binary_np,dtype=np.float32)
    count=binary_np.sum(1).astype(np.float32).reshape(-1,1)
    types=np.zeros((len(train),5),dtype=np.float32)
    exact=[[] for _ in range(len(train))]
    documents=[[] for _ in range(len(train))]
    cells=train[genes].stack().dropna()
    cells=cells[cells.ne('WT')].astype(str)
    for (row,gene),value in cells.items():
        # Original NLP documents, NOT the exp63 candidate.
        documents[row].append(f'{gene}_{value}')
        for token in value.split():
            types[row,mutation_type(token)]+=1
            match=re.search(r'(\d+)',token)
            if match:
                exact[row].append(f'{gene}_POS{int(match.group(1))}')
    hasher=FeatureHasher(n_features=2**15,input_type='string',alternate_sign=False)
    base=sparse.hstack([binary,sparse.csr_matrix(count),sparse.csr_matrix(types),
        hasher.transform(exact)],format='csr')
    docs=np.array([' '.join(t) for t in documents],dtype=object)
    return binary,base,docs

def fit_base_pair(binary,base,docs,y,fit_idx,predict_idx,forbidden,seed,nclasses):
    assert not np.intersect1d(fit_idx,predict_idx).size
    assert not np.intersect1d(fit_idx,forbidden).size
    st,sv=crossfit_signature(binary,y,fit_idx,predict_idx,seed,nclasses)
    xt=sparse.hstack([base[fit_idx],sparse.csr_matrix(st)],format='csr')
    xv=sparse.hstack([base[predict_idx],sparse.csr_matrix(sv)],format='csr')
    counts=np.bincount(y[fit_idx],minlength=nclasses).astype(np.float64)
    assert (counts>0).all()
    weight=1/np.sqrt(counts[y[fit_idx]])
    weight/=weight.mean()
    xgb=model(seed,3)
    xgb.fit(xt,y[fit_idx],sample_weight=weight)
    xp=xgb.predict_proba(xv)
    npred=original_nlp_predict(docs,fit_idx,predict_idx,y,seed)
    assert np.array_equal(xgb.classes_,np.arange(nclasses))
    for p in (xp,npred):
        assert p.shape==(len(predict_idx),nclasses)
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=2e-5)
    return xp,npred

def run(seed):
    start=time.monotonic()
    out=ROOT/f'exp_66_nested_stacker_seed{seed}'
    out.mkdir(exist_ok=False)
    train_path=ROOT/'train.csv'
    prob_path=ROOT/f'exp_56_seed{seed}_class_weight_oof_probabilities.csv'
    labels_path=ROOT/f'exp_56_seed{seed}_class_weight_predictions.csv'
    helper_paths=[ROOT/'exp_65_model_capacity_screen.py',ROOT/'exp_63_nlp_event_gene_screen.py']
    manifest=dict(seed=seed,outer_folds=5,meta_inner_folds=3,signature_inner_folds=5,
        combiner='StandardScaler + LogisticRegression(C=1,class_weight=balanced,max_iter=2000)',
        inputs_to_combiner='52 raw probabilities, XGB then NLP, sorted class order',
        meta_training='fresh inner OOF inside each outer training partition only',
        stored_oof_usage='outer validation inputs and fixed baseline only; no fitting',
        unchanged='depth3 weighted multi-event XGB, original NLP, signature protocol',
        gate_delta=.005,gate_improved_folds=4,external_data=False,evaluation_input_access=False,
        teammate_code_used=False,search=False,python=platform.python_version(),
        sklearn=sklearn.__version__,source_sha256=digest(Path(__file__)),
        file_sha256={p.name:digest(p) for p in [train_path,prob_path,labels_path,*helper_paths]})
    with (out/'manifest.json').open('x') as f:
        json.dump(manifest,f,indent=2)
    train=pd.read_csv(train_path).reset_index(drop=True)
    enc=LabelEncoder().fit(train.SUBCLASS)
    classes=enc.classes_
    y=enc.transform(train.SUBCLASS)
    saved=align(pd.read_csv(prob_path),train)
    labels=align(pd.read_csv(labels_path),train)
    xgb=saved[[f'candidate_xgb_prob_{c}' for c in classes]].to_numpy()
    nlp=saved[[f'nlp_prob_{c}' for c in classes]].to_numpy()
    baseline=saved[[f'candidate_ensemble_prob_{c}' for c in classes]].to_numpy()
    assert np.allclose(baseline,.8*xgb+.2*nlp,atol=2e-7)
    assert np.array_equal(classes[baseline.argmax(1)],labels.candidate_ensemble_pred)
    binary,base,docs=features(train)
    candidate=np.zeros_like(baseline)
    fold_index=np.zeros(len(y),dtype=int)
    results=[]
    outer=StratifiedKFold(5,shuffle=True,random_state=seed)
    print(f'exp66 seed={seed}, 5 outer x 3 inner, TRAIN only, fixed combiner',flush=True)
    for fold,(tr,va) in enumerate(outer.split(np.zeros(len(y)),y),1):
        tick=time.monotonic()
        assert np.array_equal(np.flatnonzero(labels.fold.to_numpy()==fold),va)
        if fold==1:
            print('Checking both original base learners on outer fold1',flush=True)
            ax,an=fit_base_pair(binary,base,docs,y,tr,va,va,seed,len(classes))
            checks=dict(xgb_max_diff=float(abs(ax-xgb[va]).max()),nlp_max_diff=float(abs(an-nlp[va]).max()))
            assert np.allclose(ax,xgb[va],atol=2e-6)
            assert np.allclose(an,nlp[va],atol=2e-6)
            with (out/'baseline_reproduction.json').open('x') as f:
                json.dump(checks,f,indent=2)
            print(f'Both base learners reproduce saved OOF: {checks}',flush=True)
        inner=StratifiedKFold(3,shuffle=True,random_state=seed+1000+fold)
        z=np.full((len(tr),2*len(classes)),np.nan)
        inner_fold=np.zeros(len(tr),dtype=int)
        coverage=np.zeros(len(tr),dtype=int)
        for inner_id,(it,iv) in enumerate(inner.split(np.zeros(len(tr)),y[tr]),1):
            inner_start=time.monotonic()
            fit_idx,predict_idx=tr[it],tr[iv]
            assert not np.intersect1d(predict_idx,va).size
            xp,npred=fit_base_pair(binary,base,docs,y,fit_idx,predict_idx,va,seed,len(classes))
            z[iv]=np.hstack([xp,npred])
            inner_fold[iv]=inner_id
            coverage[iv]+=1
            print(f'Outer {fold}/5 inner {inner_id}/3 done ({time.monotonic()-inner_start:.1f}s)',flush=True)
        assert (coverage==1).all() and np.isfinite(z).all()
        inner_df=pd.DataFrame(z,columns=[f'{m}_prob_{c}' for m in ('xgb','nlp') for c in classes])
        inner_df.insert(0,'inner_fold',inner_fold)
        inner_df.insert(0,'row_index',tr)
        inner_df.to_csv(out/f'outer{fold}_meta_training_probabilities.csv',index=False,mode='x')
        meta=make_pipeline(StandardScaler(),LogisticRegression(C=1.0,class_weight='balanced',
            solver='lbfgs',max_iter=2000,random_state=seed))
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning)
            meta.fit(z,y[tr])
        # Only here do saved outer validation predictions enter the combiner.
        p=meta.predict_proba(np.hstack([xgb[va],nlp[va]]))
        assert np.array_equal(meta.classes_,np.arange(len(classes)))
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
        candidate[va]=p
        fold_index[va]=fold
        joblib.dump(meta,out/f'outer{fold}_combiner.joblib',compress=3)
        bscore=float(f1_score(y[va],baseline[va].argmax(1),average='macro',zero_division=0))
        cscore=float(f1_score(y[va],p.argmax(1),average='macro',zero_division=0))
        row=dict(seed=seed,fold=fold,baseline_f1=bscore,candidate_f1=cscore,
            delta=cscore-bscore,seconds=time.monotonic()-tick)
        results.append(row)
        pd.DataFrame(p,columns=classes).to_csv(out/f'fold{fold}_probabilities.csv',index=False,mode='x')
        print(json.dumps(row),flush=True)
    bp,cp=baseline.argmax(1),candidate.argmax(1)
    summary=dict(seed=seed,baseline_f1=float(f1_score(y,bp,average='macro')),
        candidate_f1=float(f1_score(y,cp,average='macro')),
        improved_folds=sum(r['delta']>0 for r in results),rescue=int(((bp!=y)&(cp==y)).sum()),
        damage=int(((bp==y)&(cp!=y)).sum()),seconds=time.monotonic()-start)
    summary['delta']=summary['candidate_f1']-summary['baseline_f1']
    summary['pass']=summary['delta']>=.005 and summary['improved_folds']>=4
    with (out/'summary.json').open('x') as f:
        json.dump(summary,f,indent=2)
    pd.DataFrame(results).to_csv(out/'fold_results.csv',index=False,mode='x')
    b=precision_recall_fscore_support(y,bp,labels=np.arange(len(classes)),zero_division=0)
    c=precision_recall_fscore_support(y,cp,labels=np.arange(len(classes)),zero_division=0)
    pd.DataFrame(dict(subclass=classes,support=b[3],baseline_f1=b[2],candidate_f1=c[2],
        delta=c[2]-b[2])).to_csv(out/'classwise.csv',index=False,mode='x')
    metadata=pd.DataFrame(dict(row_index=np.arange(len(y)),ID=train.ID,true_subclass=train.SUBCLASS,
        seed=seed,fold=fold_index,baseline_pred=classes[bp],candidate_pred=classes[cp]))
    pd.concat([metadata,pd.DataFrame(baseline,columns=[f'baseline_prob_{c}' for c in classes]),
        pd.DataFrame(candidate,columns=[f'candidate_prob_{c}' for c in classes])],axis=1).to_csv(
            out/'oof_probabilities.csv',index=False,mode='x')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=19,choices=[19,193,2029])
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.seed)
