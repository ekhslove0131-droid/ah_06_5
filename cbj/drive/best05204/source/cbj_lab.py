"""One Linux runner for Windows WSL experiments and later clean Colab replay.

Current conditions: unchanged full MCMP, matched-size F relevance, mRMR.
No test, submission, server feedback, meta-model, or continuous parameter search.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import psutil
from scipy import sparse
from sklearn.feature_selection import f_classif
import xgboost as xgb
from mrmr import mrmr_classif

import mcmp_repr as M
import lab_metrics as L


CONDITIONS = ('full', 'relevance', 'mrmr')


def emit(out, event, **values):
    row = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),event=event,**values)
    text = json.dumps(row,ensure_ascii=False,allow_nan=False)
    print(text,flush=True)
    with (Path(out)/'progress.jsonl').open('a') as stream:
        stream.write(text+'\n')


def write_gzip(path, value):
    partial = Path(str(path)+'.partial')
    with gzip.open(partial,'wt',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False)
    partial.replace(path)


def read_gzip(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:
        return json.load(stream)


def manifest(out, signature, files):
    L.write_json(Path(out)/'COMPLETE.json',dict(signature=signature,
        files={name:L.sha(Path(out)/name) for name in files}))


def valid(out, signature):
    path=Path(out)/'COMPLETE.json'
    if not path.is_file():
        return False
    receipt=L.read_json(path)
    L.need(receipt['signature']==signature,'Existing artifact belongs to another configuration: '+str(out))
    L.need(all((Path(out)/name).is_file() and L.sha(Path(out)/name)==digest
               for name,digest in receipt['files'].items()),'Artifact integrity failure: '+str(out))
    return True


def check_inputs(config, inputs):
    L.need(config['version']=='cbj-windows-lab-v1','Wrong configuration version.')
    L.need(set(config['inputs'])=={'train','reference'},'Only training data and original OOF are allowed.')
    L.need(not any(k in config for k in ('test','server_scores','bias','optuna')),'Unsupported data/tuning input.')
    for name,digest in config['source_hashes'].items():
        L.need(Path(name).name==name and L.sha(Path(__file__).parent/name)==digest,'Source hash mismatch: '+name)
    for name,spec in config['inputs'].items():
        L.need(Path(spec['filename']).name==spec['filename'],'Unsafe input path.')
        L.need(L.sha(Path(inputs)/spec['filename'])==spec['sha256'],'Input hash mismatch: '+name)
    a=L.load_oof(Path(inputs)/config['inputs']['reference']['filename'])
    e=config['expected']
    L.need(len(a['ids'])==e['rows'] and a['class_order'].tolist()==e['class_order'],'Reference schema mismatch.')
    L.need(np.unique(a['folds']).tolist()==e['folds'],'Reference folds changed.')
    return a


def environment(config, training):
    versions={name:importlib.metadata.version(name) for name in config['packages']}
    L.need(versions==config['packages'],'Package versions differ from the locked experiment.')
    result=dict(python=platform.python_version(),platform=platform.platform(),packages=versions,
                gpu=None,training_requested=training)
    if training:
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,memory.free',
                                      '--format=csv,noheader,nounits'],text=True)
        rows=[]
        for line in text.strip().splitlines():
            idx,name,uid,driver,total,free=[v.strip() for v in line.split(',')]
            rows.append(dict(index=int(idx),name=name,uuid=uid,driver=driver,total_mib=int(total),free_mib=int(free)))
        found=[r for r in rows if r['index']==config['runtime']['gpu_index']]
        L.need(len(found)==1 and found[0]['free_mib']>=config['runtime']['minimum_free_mib'],'GPU unavailable/insufficient free VRAM.')
        L.need(xgb.build_info().get('USE_CUDA') is True,'XGBoost CUDA build required; no CPU fallback.')
        result['gpu']=found[0]
    return result


def prepare(config, inputs, out, a):
    signature=L.objsha(dict(inputs=config['inputs'],representation=config['representation'],
                           source_genes=config['source_genes'],parser=config['source_hashes']['mcmp_repr.py']))
    prepared=Path(out)/'prepared'
    prepared.mkdir(exist_ok=True)
    folds=config['expected']['folds']
    if all(valid(prepared/f'fold_{fold}',L.objsha([signature,fold])) for fold in folds):
        emit(out,'PREPARED_REUSE',folds=len(folds))
        return
    train=pd.read_csv(Path(inputs)/config['inputs']['train']['filename'],dtype=str)
    L.need(train.columns.tolist()==['ID','SUBCLASS']+config['source_genes'],'Raw schema or gene order mismatch.')
    L.need(train.ID.tolist()==a['ids'].tolist(),'Train/OOF ID order mismatch.')
    L.need(train.SUBCLASS.tolist()==a['class_order'][a['y']].tolist(),'Train/OOF labels mismatch.')
    emit(out,'PARSER_START',rows=len(train),genes=len(config['source_genes']))
    rows,parser_stats=M.build_feature_rows(train,config['source_genes'])
    del train
    gc.collect()
    for fold in folds:
        job=prepared/f'fold_{fold}'
        job.mkdir(exist_ok=True)
        sig=L.objsha([signature,fold])
        if valid(job,sig):
            continue
        tr=np.flatnonzero(a['folds']!=fold)
        va=np.flatnonzero(a['folds']==fold)
        L.need(not set(a['groups'][tr]) & set(a['groups'][va]),'Canonical leakage.')
        encoder=M.Encoder(config['representation']['blocks'],config['representation']['min_group_support'])
        xt=encoder.fit_transform([rows[i] for i in tr],a['groups'][tr])
        xv=encoder.transform([rows[i] for i in va])
        spec=encoder.to_dict()
        restored=M.Encoder.from_dict(spec)
        L.need(np.array_equal(xv[:3].toarray(),restored.transform([rows[i] for i in va[:3]]).toarray()),'Encoder reload mismatch.')
        sparse.save_npz(job/'train.npz',xt)
        sparse.save_npz(job/'valid.npz',xv)
        write_gzip(job/'encoder.json.gz',spec)
        L.save_npz(job/'identity.npz',train_indices=tr,valid_indices=va)
        L.write_json(job/'review.json',dict(fold=fold,train_rows=len(tr),valid_rows=len(va),features=len(encoder.names),
            original_genes=len(config['source_genes']),names_sha256=L.objsha(encoder.names),
            train_ids_sha256=L.objsha(a['ids'][tr].tolist()),valid_ids_sha256=L.objsha(a['ids'][va].tolist()),
            fit_scope='fold training patients only',parser_stats=parser_stats))
        manifest(job,sig,['train.npz','valid.npz','encoder.json.gz','identity.npz','review.json'])
        emit(out,'FOLD_PREPARED',fold=fold,features=len(encoder.names))
        del xt,xv,encoder,restored
        gc.collect()
    del rows
    gc.collect()


def selection(config, xt, names, y, tr_ids, job, fold, out):
    rule=config['selection']
    sig=L.objsha(dict(rule=rule,names=names,training_ids=tr_ids,source=config['source_hashes']['cbj_lab.py']))
    if valid(job,sig):
        with np.load(job/'masks.npz',allow_pickle=False) as z:
            return {k:z[k] for k in z.files}
    job.mkdir(parents=True,exist_ok=True)
    candidate=np.array([i for i,n in enumerate(names) if n.startswith(tuple(rule['candidate_prefixes']))],dtype=int)
    candidate_set=set(candidate)
    protected=np.array([i for i in range(len(names)) if i not in candidate_set],dtype=int)
    x=xt[:,candidate].copy()
    x.sum_duplicates();x.eliminate_zeros()
    L.need(np.all(x.data==1),'Candidate columns must be binary.')
    count=np.asarray(x.sum(0)).ravel()
    active=(count>0)&(count<len(y))
    candidate=candidate[active];x=x[:,active]
    L.need(len(candidate)>0,'No variable derived candidates.')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        relevance=f_classif(x,y)[0]
    relevance=np.nan_to_num(relevance,nan=0,posinf=np.finfo(float).max,neginf=0)
    order=np.lexsort((candidate,-relevance))
    pool=candidate[order[:min(rule['prefilter_max'],len(candidate))]]
    k=min(rule['selected_k'],len(pool))
    relevance_indices=pool[:k]
    emit(out,'MRMR_START',fold=fold,candidates=len(pool),selected_k=k,selector_device='CPU',new_classifier_fits=0)
    started=time.perf_counter()
    frame=pd.DataFrame(xt[:,pool].toarray(),columns=[str(i) for i in pool])
    selected=mrmr_classif(X=frame,y=pd.Series(y),K=k,n_jobs=rule['selector_jobs'],
                         relevance='f',redundancy='c',denominator='mean',show_progress=False)
    chosen=np.array([int(i) for i in selected],dtype=int)
    L.need(len(set(chosen))==k and set(chosen)<=set(pool),'Invalid mRMR selection.')
    masks=dict(full=np.arange(len(names)),relevance=np.sort(np.r_[protected,relevance_indices]),
               mrmr=np.sort(np.r_[protected,chosen]))
    L.need(len(masks['relevance'])==len(masks['mrmr']),'Unmatched selection sizes.')
    L.save_npz(job/'masks.npz',**masks)
    pool_scores={int(i):float(s) for i,s in zip(candidate,relevance)}
    relevance_set,chosen_set=set(relevance_indices),set(chosen)
    pd.DataFrame([dict(original_index=int(i),feature=names[i],f_relevance=pool_scores[int(i)],
        relevance_selected=bool(i in relevance_set),mrmr_selected=bool(i in chosen_set)) for i in pool]).to_csv(job/'candidate_scores.csv',index=False)
    for condition,indices in masks.items():
        keep=set(indices)
        write_gzip(job/f'{condition}_feature_mask.json.gz',dict(original_names_sha256=L.objsha(names),
            selected_names=[names[i] for i in indices],selected_indices=indices.tolist(),
            dropped_names=[name for i,name in enumerate(names) if i not in keep],
            policy=('all original fold-fitted features' if condition=='full' else
                    'protect other blocks; variable binary candidate pool by train F; '+condition)))
    L.write_json(job/'review.json',dict(fold=fold,fit_ids_sha256=L.objsha(tr_ids),candidate_count=len(candidate),
        pool_size=len(pool),k=k,protected=len(protected),original_features=len(names),
        selected_features={key:len(value) for key,value in masks.items()},seconds=time.perf_counter()-started,
        train_only=True,test_read=False,selector_device='CPU',classifier_device='GPU',
        implementation='mrmr-selection 0.2.8 F relevance / mean absolute correlation redundancy; input pool order is fixed'))
    manifest(job,sig,['masks.npz','candidate_scores.csv','review.json']+[f'{c}_feature_mask.json.gz' for c in CONDITIONS])
    emit(out,'MRMR_COMPLETE',fold=fold,seconds=time.perf_counter()-started)
    return masks


def predict(model, xv, take, batch):
    result=[]
    for start in range(0,xv.shape[0],batch):
        dense=xv[start:start+batch,take].toarray(order='C')
        result.append(L.prob(model.get_booster().predict(xgb.DMatrix(dense),strict_shape=True)))
    return np.concatenate(result)


def train_one(config, xt, xv, take, y, job, fold, condition, env, out, names):
    sig=L.objsha(dict(model=config['model'],columns=take.tolist(),names_sha256=L.objsha(names),
        fold=fold,condition=condition,inputs=config['inputs'],sources=config['source_hashes'],packages=env['packages']))
    job.mkdir(parents=True,exist_ok=True)
    if valid(job,sig):
        with np.load(job/'valid_prob.npz',allow_pickle=False) as z:
            return L.prob(z['prob'])
    environment(config,True)
    rt=config['runtime']
    dense_bytes=xt.shape[0]*len(take)*4
    estimated=psutil.Process().memory_info().rss+dense_bytes*rt['dense_multiplier']+2**30
    L.need(estimated<rt['max_ram_gib']*2**30 and dense_bytes*rt['dense_multiplier']<.8*psutil.virtual_memory().available,
           'Insufficient RAM; no change in input representation allowed.')
    dense=xt[:,take].toarray(order='C')
    L.need(dense.dtype==np.float32 and np.isfinite(dense).all(),'Invalid numerical features.')
    params=dict(config['model'])
    params.update(device=f"cuda:{rt['gpu_index']}",n_jobs=rt['threads'],num_class=len(config['expected']['class_order']))
    model=xgb.XGBClassifier(**params)
    emit(out,'GPU_FIT_START',fold=fold,condition=condition,rows=len(y),features=len(take))
    started=time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(dense,y)
    fit_seconds=time.perf_counter()-started
    effective=json.loads(model.get_booster().save_config())
    L.need(effective['learner']['generic_param']['device'].startswith('cuda'),'Training did not use CUDA.')
    L.need(model.get_booster().num_boosted_rounds()==params['n_estimators'],'Tree budget changed.')
    L.need(model.classes_.tolist()==list(range(params['num_class'])),'Model class order mismatch.')
    del dense
    gc.collect()
    prediction=predict(model,xv,take,rt['predict_batch_rows'])
    model.save_model(job/'model.ubj')
    restored=xgb.XGBClassifier()
    restored.load_model(job/'model.ubj')
    probe=predict(restored,xv[:min(16,xv.shape[0])],take,rt['predict_batch_rows'])
    L.need(np.max(abs(probe-prediction[:len(probe)]))<2e-6,'Reload prediction parity failed.')
    L.save_npz(job/'valid_prob.npz',prob=prediction)
    L.save_npz(job/'columns.npz',indices=take)
    L.write_json(job/'review.json',dict(fold=fold,condition=condition,fit_seconds=fit_seconds,
        features=len(take),effective_parameters=effective,requested_parameters=params,
        gpu=env['gpu'],warnings=[str(w.message) for w in captured],estimated_ram_bytes=estimated,
        reload_max_abs=float(np.max(abs(probe-prediction[:len(probe)])))))
    manifest(job,sig,['model.ubj','valid_prob.npz','columns.npz','review.json'])
    emit(out,'GPU_FIT_COMPLETE',fold=fold,condition=condition,fit_seconds=fit_seconds)
    del model,restored
    gc.collect()
    return prediction


def report(config, a, predictions, out, mode):
    results={key:L.metrics(a,p) for key,p in predictions.items()}
    reference=L.metrics(a,a['prob'])
    table=[]
    for name,s in results.items():
        table.append(dict(condition=name,**{k:v for k,v in s.items() if k not in ('per_class','fold_f1')},
            **{f'fold_{i}':v for i,v in zip(config['expected']['folds'],s['fold_f1'])},
            delta_vs_saved_reference=s['macro_f1']-reference['macro_f1']))
        L.save_npz(Path(out)/f'{name}_oof.npz',**{k:a[k] for k in ('ids','y','groups','folds','class_order')},prob=predictions[name])
        pd.DataFrame(s['per_class']).to_csv(Path(out)/f'{name}_per_class.csv',index=False)
    pd.DataFrame(table).to_csv(Path(out)/f'{mode}_scoreboard.csv',index=False)
    comparisons={name:dict(**L.changes(a['y'],p,a['prob']),
         bootstrap=L.bootstrap(a,p,a['prob'],config['bootstrap_resamples'],42)) for name,p in predictions.items()}
    if 'full' in predictions:
        comparisons['full_saved_reference_parity']=dict(max_probability_abs=float(np.max(abs(predictions['full']-a['prob']))),
            top1_disagreement=float((predictions['full'].argmax(1)!=a['prob'].argmax(1)).mean()))
        for name in ('relevance','mrmr'):
            if name in predictions:
                comparisons[name+'_vs_windows_full']=dict(**L.changes(a['y'],predictions[name],predictions['full']),
                    delta_macro_f1=results[name]['macro_f1']-results['full']['macro_f1'],
                    bootstrap=L.bootstrap(a,predictions[name],predictions['full'],config['bootstrap_resamples'],42))
    if 'mrmr' in predictions and 'relevance' in predictions:
        comparisons['mrmr_vs_same_size_relevance']=dict(delta_macro_f1=results['mrmr']['macro_f1']-results['relevance']['macro_f1'],
            **L.changes(a['y'],predictions['mrmr'],predictions['relevance']))
    value=dict(status='DEVELOPMENT_BASE_CV_COMPLETE',mode=mode,summaries=results,reference=reference,comparisons=comparisons,
        same_canonical_folds=True,whole_stack_nested_verified=False,test_read=False,server_score_used=False,
        submission_created=False,automatic_promotion=False,adaptive_development_selection=True)
    L.write_json(Path(out)/f'{mode}_review.json',value)
    emit(out,'RUN_COMPLETE',mode=mode,scores={k:s['macro_f1'] for k,s in results.items()})
    return value


def run(config, inputs, out, mode):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    a=check_inputs(config,inputs)
    env=environment(config,mode!='audit')
    L.write_json(out/f'environment_{mode}.json',env)
    signature=L.objsha(config)
    contract=out/'CONTRACT.json'
    if contract.exists():
        L.need(L.read_json(contract)['signature']==signature,'Use a new output directory for changed experiments.')
    else:
        L.write_json(contract,dict(signature=signature,config=config,scope='canonical grouped development base CV'))
    prepare(config,inputs,out,a)
    if mode=='audit':
        L.write_json(out/'AUDIT_COMPLETE.json',dict(status='AUDIT_COMPLETE',rows=len(a['ids']),new_fits=0,test_read=False))
        emit(out,'AUDIT_COMPLETE',new_fits=0)
        return
    conditions=['full'] if mode=='baseline' else list(CONDITIONS)
    predictions={name:np.full_like(a['prob'],np.nan) for name in conditions}
    seen=np.zeros(len(a['ids']),int)
    for fold in config['expected']['folds']:
        prep=out/'prepared'/f'fold_{fold}'
        xt=sparse.load_npz(prep/'train.npz')
        xv=sparse.load_npz(prep/'valid.npz')
        names=read_gzip(prep/'encoder.json.gz')['names']
        with np.load(prep/'identity.npz',allow_pickle=False) as identity:
            tr,va=identity['train_indices'],identity['valid_indices']
        L.need(np.array_equal(va,np.flatnonzero(a['folds']==fold)),'Prepared identity mismatch.')
        masks={'full':np.arange(len(names))}
        if mode=='experiment':
            masks=selection(config,xt,names,a['y'][tr],a['ids'][tr].tolist(),out/'selection'/f'fold_{fold}',fold,out)
        for condition in conditions:
            predictions[condition][va]=train_one(config,xt,xv,masks[condition],a['y'][tr],
                out/'models'/condition/f'fold_{fold}',fold,condition,env,out,names)
        seen[va]+=1
        del xt,xv
        gc.collect()
    L.need(np.all(seen==1),'OOF coverage must be exactly once.')
    check_inputs(config,inputs)
    value=report(config,a,predictions,out,mode)
    L.write_json(out/f'{mode.upper()}_COMPLETE.json',dict(status=value['status'],
        review_sha256=L.sha(out/f'{mode}_review.json'),conditions=conditions,
        oof_sha256={name:L.sha(out/f'{name}_oof.npz') for name in conditions}))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--mode',choices=['audit','baseline','experiment'],default='audit')
    args=parser.parse_args()
    try:
        run(L.read_json(args.config),args.inputs,args.out,args.mode)
    except Exception as exc:
        L.write_json(Path(args.out)/'FAILED.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':
    main()
