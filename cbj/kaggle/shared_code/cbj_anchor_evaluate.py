"""Train the frozen anchor and assess promotion; no full refit or registry write."""
import argparse,gzip,hashlib,json,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score,accuracy_score,balanced_accuracy_score,log_loss,classification_report
import joblib


def probability_diagnostics(y,p,n_bins=10):
    p=np.asarray(p,dtype=float);y=np.asarray(y,dtype=int);k=p.shape[1]
    assert np.isfinite(p).all() and (p>=0).all() and np.allclose(p.sum(axis=1),1,atol=1e-5)
    pred=p.argmax(axis=1);conf=p.max(axis=1);correct=pred==y
    sorted_p=np.sort(p,axis=1)
    entropy=-(p*np.log(np.clip(p,1e-15,1))).sum(axis=1)
    brier=np.mean((p*p).sum(axis=1)-2*p[np.arange(len(y)),y]+1)
    bins=np.minimum((conf*n_bins).astype(int),n_bins-1);rows=[];ece=0.
    for b in range(n_bins):
        ix=bins==b;n=int(ix.sum())
        if n:
            c=float(conf[ix].mean());a=float(correct[ix].mean());ece+=n/len(y)*abs(c-a)
            rows.append({'bin':b,'lower':b/n_bins,'upper':(b+1)/n_bins,'n':n,'mean_confidence':c,'accuracy':a})
    h=conf>=.9
    summary={'brier_multiclass_0_to_2':float(brier),'toplabel_ece_10_equal_width_bins':float(ece),
             'normalized_entropy_mean':float(np.mean(entropy/np.log(k))),
             'confidence_mean':float(conf.mean()),'top1_margin_mean':float((sorted_p[:,-1]-sorted_p[:,-2]).mean()),
             'high_confidence_09_n':int(h.sum()),'high_confidence_09_accuracy':float(correct[h].mean()) if h.any() else None}
    detail=pd.DataFrame({'predicted_index':pred,'confidence':conf,'correct':correct,'normalized_entropy':entropy/np.log(k),
                         'effective_class_count':np.exp(entropy),'top1_margin':sorted_p[:,-1]-sorted_p[:,-2],
                         'probability_variance_across_classes':p.var(axis=1),'true_class_probability':p[np.arange(len(y)),y]})
    return summary,pd.DataFrame(rows),detail


def metrics(y,p,k):
    pred=p.argmax(axis=1)
    return {'macro_f1':float(f1_score(y,pred,labels=np.arange(k),average='macro',zero_division=0)),
            'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'log_loss':float(log_loss(y,p,labels=np.arange(k)))}


def fit_checked(x,y,params,name,out):
    print('FIT START',name,x.shape,flush=True);t=time.time();model=LogisticRegression(**params)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always',ConvergenceWarning);model.fit(x,y)
    info={'name':name,'converged':not any(issubclass(w.category,ConvergenceWarning) for w in ws),
          'iterations':int(model.n_iter_.max()),'seconds':time.time()-t,'finite_coefficients':bool(np.isfinite(model.coef_).all())}
    joblib.dump(model,out/(name+'.joblib'),compress=3)
    print('FIT END',json.dumps(info),flush=True)
    return model,info


def paired_group_bootstrap(y,model_pred,reference_pred,groups,k,reps=1000,seed=4201):
    group_index=pd.factorize(groups,sort=True)[0];ng=group_index.max()+1;rng=np.random.default_rng(seed);values=[]
    for _ in range(reps):
        group_weight=np.bincount(rng.integers(0,ng,ng),minlength=ng);w=group_weight[group_index]
        scores=[]
        for pred in [model_pred,reference_pred]:
            cm=np.bincount(y*k+pred,weights=w,minlength=k*k).reshape(k,k)
            den=cm.sum(axis=0)+cm.sum(axis=1)
            scores.append(np.divide(2*np.diag(cm),den,out=np.zeros(k),where=den>0).mean())
        values.append([scores[0],scores[0]-scores[1]])
    a=np.array(values)
    return {'replicates':reps,'unit':'canonical profile group','conditional_on_fitted_OOF_models':True,
            'anchor_macro_f1_ci95':np.quantile(a[:,0],[.025,.975]).tolist(),
            'delta_macro_f1_ci95':np.quantile(a[:,1],[.025,.975]).tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',action='store_true');ap.add_argument('--prepared',type=Path,required=True);ap.add_argument('--policy',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    if not args.train or not Path('/kaggle/working').exists():raise SystemExit('Explicit --train and Kaggle runtime required')
    root=args.prepared;out=args.output;out.mkdir(exist_ok=True,parents=True)
    policy=json.loads(args.policy.read_text());pre=json.loads((root/'preflight.json').read_text());cfg=json.loads((root/'anchor_config.json').read_text())
    assert hashlib.sha256((root/'preflight.json').read_bytes()).hexdigest()==policy['expected_preflight_sha256']
    for name,h in pre['file_sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
    (out/'promotion_policy.json').write_text(json.dumps(policy,indent=2))
    folds=pd.read_csv(root/'fold_assignments.csv');classes=cfg['class_order'];k=len(classes);y=folds.SUBCLASS.map({c:i for i,c in enumerate(classes)}).to_numpy();n=len(y)
    assert folds.groupby('canonical_hash').fold.nunique().eq(1).all()
    allp={name:np.full((n,k),np.nan) for name in ['anchor','prior','burden','gene_only']};fit_reports=[];fold_rows=[]
    for fold in range(cfg['n_folds']):
        tr=np.flatnonzero(folds.fold.ne(fold));va=np.flatnonzero(folds.fold.eq(fold))
        assert set(folds.canonical_hash.iloc[tr]).isdisjoint(folds.canonical_hash.iloc[va])
        with gzip.open(root/f'fold_{fold}_encoder.json.gz','rt') as f:enc=json.load(f)
        assert set(enc['fit_ids'])==set(folds.ID.iloc[tr])
        xt=load_npz(root/f'fold_{fold}_train.npz');xv=load_npz(root/f'fold_{fold}_valid.npz')
        prior=(np.bincount(y[tr],minlength=k)+1)/(len(tr)+k);allp['prior'][va]=prior
        cols={'anchor':None,'burden':[enc['names'].index('global|burden_log')],
              'gene_only':[i for i,name in enumerate(enc['names']) if name.startswith('gene|') and name.endswith('|nonWT')]}
        for name,indices in cols.items():
            a=xt if indices is None else xt[:,indices];b=xv if indices is None else xv[:,indices]
            model,info=fit_checked(a,y[tr],cfg['model_params'],f'{name}_fold_{fold}',out)
            assert list(model.classes_)==list(range(k));p=model.predict_proba(b);allp[name][va]=p;info['fold']=fold;info['role']=name;fit_reports.append(info)
            row={'fold':fold,'role':name,'train_n':len(tr),'valid_n':len(va),**metrics(y[va],p,k)};fold_rows.append(row);print('SCORES',json.dumps(row),flush=True)
        fold_rows.append({'fold':fold,'role':'prior','train_n':len(tr),'valid_n':len(va),**metrics(y[va],allp['prior'][va],k)})
        pd.DataFrame(fold_rows).to_csv(out/'fold_metrics.csv',index=False)
        (out/'fit_status.json').write_text(json.dumps(fit_reports,indent=2))
    evaluations={};diag_table=None
    novel=pd.read_csv(root/'oof_novelty_by_patient.csv').set_index('ID').loc[folds.ID].reset_index()
    for name,p in allp.items():
        d,cal,detail=probability_diagnostics(y,p);evaluations[name]={**metrics(y,p,k),**d}
        table=folds.copy()
        for j,cl in enumerate(classes):table['p_'+cl]=p[:,j]
        if name=='anchor':
            for col in novel:
                if col not in table:table[col]=novel[col].values
            diag_table=pd.concat([table,detail],axis=1)
        table.to_csv(out/(name+'_oof_probabilities.csv'),index=False)
        cal.to_csv(out/(name+'_calibration.csv'),index=False)
    diag_table.to_csv(out/'anchor_uncertainty_by_patient.csv',index=False)
    pd.DataFrame(classification_report(y,allp['anchor'].argmax(axis=1),labels=np.arange(k),target_names=classes,output_dict=True,zero_division=0)).T.to_csv(out/'per_class.csv')
    fm=pd.DataFrame(fold_rows);fs=fm.loc[fm.role=='anchor','macro_f1'].to_numpy()
    variation={'fold_f1_mean':float(fs.mean()),'fold_f1_std_ddof1':float(fs.std(ddof=1)),
               'fold_f1_variance_ddof1':float(fs.var(ddof=1)),'fold_f1_min':float(fs.min()),'fold_f1_max':float(fs.max()),'fold_f1_range':float(np.ptp(fs)),
               'note':'Folds are not independent repetitions; this measures split variability, not all training randomness.'}
    boot={name:paired_group_bootstrap(y,allp['anchor'].argmax(axis=1),allp[name].argmax(axis=1),folds.canonical_hash,k,policy['bootstrap_replicates']) for name in ['burden','gene_only']}
    # Dispersion is in comparable 26-class OOF probability space, not raw gene coordinates.
    p=allp['anchor'];center=p.mean(axis=0);within=between=0.;dispersion=[]
    for cl in range(k):
        z=p[y==cl];mean=z.mean(axis=0);w=float(((z-mean)**2).sum()/len(z));b=float(((mean-center)**2).sum())
        within+=len(z)*w/n;between+=len(z)*b/n
        dispersion.append({'SUBCLASS':classes[cl],'n':len(z),'within_probability_variance_trace':w,'centroid_distance_squared_to_global':b,
                           'mean_entropy':float(diag_table.loc[y==cl,'normalized_entropy'].mean()),'mean_confidence':float(diag_table.loc[y==cl,'confidence'].mean())})
    pd.DataFrame(dispersion).to_csv(out/'class_dispersion.csv',index=False)
    variation.update(within_class_probability_variance=within,between_class_probability_variance=between,
                     between_fraction_of_total=between/(within+between) if within+between else None)
    # Natural novelty cohorts with per-class outcomes; zero-token rows are separate.
    masks={'no_variant':novel.exact_n.eq(0),'known_exact_only':novel.exact_n.gt(0)&novel.exact_unseen.eq(0),
           'any_unseen_exact':novel.exact_unseen.gt(0),'mostly_unseen_exact':novel.unseen_exact_fraction.ge(.5),'any_unseen_site':novel.site_unseen.gt(0)}
    subgroup=[]
    for group,mask in masks.items():
        for cl in range(k):
            ix=np.flatnonzero(mask.to_numpy()&(y==cl))
            if len(ix):subgroup.append({'group':group,'SUBCLASS':classes[cl],'n':len(ix),'recall':float(np.mean(p[ix].argmax(axis=1)==cl)),
                                       'mean_true_class_log_loss':float(-np.log(np.clip(p[ix,cl],1e-15,1)).mean())})
    pd.DataFrame(subgroup).to_csv(out/'novelty_class_metrics.csv',index=False)
    # Cold-site model uses exactly the saved split and train-only encoder.
    cold=pd.read_csv(root/'cold_site_split.csv');cy=cold.SUBCLASS.map({c:i for i,c in enumerate(classes)}).to_numpy()
    tr=np.flatnonzero(cold.split.eq('train'));va=np.flatnonzero(cold.split.eq('cold_validation'));present=sorted(set(cy[va]))
    with gzip.open(root/'cold_encoder.json.gz','rt') as f:enc=json.load(f)
    assert 'site|BRAF|600' not in enc['seen']['site'];assert set(enc['fit_ids'])==set(cold.ID.iloc[tr])
    xt=load_npz(root/'cold_train.npz');xv=load_npz(root/'cold_valid.npz');coldp={};cold_metrics={}
    prior=(np.bincount(cy[tr],minlength=k)+1)/(len(tr)+k);coldp['prior']=np.tile(prior,(len(va),1))
    for name,indices in [('anchor',None),('burden',[enc['names'].index('global|burden_log')])]:
        model,info=fit_checked(xt if indices is None else xt[:,indices],cy[tr],cfg['model_params'],'cold_'+name,out)
        info['role']='cold_'+name;fit_reports.append(info);coldp[name]=model.predict_proba(xv if indices is None else xv[:,indices])
    for name,cp in coldp.items():
        d,cal,detail=probability_diagnostics(cy[va],cp)
        cold_metrics[name]={**metrics(cy[va],cp,k),**d,'macro_f1_present_classes':float(f1_score(cy[va],cp.argmax(axis=1),labels=present,average='macro',zero_division=0))}
        table=cold.iloc[va].copy().reset_index(drop=True)
        for j,cl in enumerate(classes):table['p_'+cl]=cp[:,j]
        table.to_csv(out/('cold_'+name+'_probabilities.csv'),index=False)
    cold_context={'train_n':len(tr),'validation_n':len(va),'present_classes':[classes[i] for i in present],
                  'note':'Restricted BRAF600 stress scenario, seven validation classes. Not directly comparable with all-class OOF.'}
    a=evaluations['anchor'];weak=max(evaluations[x]['macro_f1'] for x in ['prior','burden'])
    all_converged=all(i['converged'] and i['finite_coefficients'] for i in fit_reports)
    gates={
        'all_fits_converged_finite':all_converged,
        'beats_weak_controls':a['macro_f1']>=weak+policy['min_delta_vs_weak_controls'],
        'burden_delta_ci_positive':boot['burden']['delta_macro_f1_ci95'][0]>0,
        'gene_only_point_noninferiority':a['macro_f1']>=evaluations['gene_only']['macro_f1']-policy['gene_only_point_tolerance'],
        'gene_only_ci_noninferiority':boot['gene_only']['delta_macro_f1_ci95'][0]>-policy['gene_only_ci_tolerance'],
        'fold_std':variation['fold_f1_std_ddof1']<=policy['max_fold_std'],
        'worst_fold':variation['fold_f1_min']>=a['macro_f1']-policy['max_worst_fold_gap'],
        'probabilistic_scores_better_than_prior':a['log_loss']<evaluations['prior']['log_loss'] and a['brier_multiclass_0_to_2']<evaluations['prior']['brier_multiclass_0_to_2'],
        'ece':a['toplabel_ece_10_equal_width_bins']<=policy['max_ece'],
        'high_confidence_accuracy':a['high_confidence_09_n']<policy['min_high_conf_n'] or a['high_confidence_09_accuracy']>=policy['min_high_conf_accuracy'],
        'cold_not_collapsed':cold_metrics['anchor']['macro_f1_present_classes']>=max(cold_metrics[x]['macro_f1_present_classes'] for x in ['prior','burden'])-policy['cold_f1_tolerance'] and cold_metrics['anchor']['log_loss']<=policy['cold_logloss_prior_multiplier']*cold_metrics['prior']['log_loss'],
    }
    gates={k:bool(v) for k,v in gates.items()}
    report={'evaluations':evaluations,'variation':variation,'paired_group_bootstrap':boot,'cold_metrics':cold_metrics,'cold_context':cold_context,
            'fit_reports':fit_reports,'promotion_gates':gates,'automatic_gates_passed':all(gates.values()),
            'registry_promotion_status':'AWAITING_GPT_AND_ARTIFACT_REVIEW' if all(gates.values()) else 'HOLD_FAILED_GATES',
            'full_train_refit_done':False,'n_oof':n,'class_order':classes,'policy_sha256':hashlib.sha256(args.policy.read_bytes()).hexdigest(),
            'preflight_sha256':policy['expected_preflight_sha256'],'new_external_data_performance':'NOT VERIFIED'}
    (out/'assessment.json').write_text(json.dumps(report,indent=2))
    (out/'fit_status.json').write_text(json.dumps(fit_reports,indent=2))
    print('ASSESSMENT',json.dumps({k:report[k] for k in ['evaluations','variation','promotion_gates','registry_promotion_status']}),flush=True)


if __name__=='__main__':main()
