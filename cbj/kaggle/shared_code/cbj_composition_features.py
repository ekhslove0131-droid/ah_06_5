"""Deterministic distribution summaries; no labels, learned ranks or severity scores."""
import numpy as np


def composition_features(matrix,names):
    index={name:i for i,name in enumerate(names)}
    pair_names=[n for n in names if n.startswith('aa|')]
    from_names=sorted(n for n in names if n.startswith('aa_from|'))
    to_names=sorted(n for n in names if n.startswith('aa_to|'))
    type_names=[n for n in names if n.startswith('global|type_fraction|')]
    if not pair_names or not type_names or [n.split('|')[1] for n in from_names]!=[n.split('|')[1] for n in to_names]:
        raise ValueError('Required composition inputs absent or misaligned')
    def distribution(keys):
        values=matrix[:,[index[k] for k in keys]].astype(np.float64)
        sums=values.sum(axis=1,keepdims=True)
        return np.divide(values,sums,out=np.zeros_like(values),where=sums>0),sums.ravel()
    def entropy(p):
        return -(p*np.log(np.clip(p,1e-15,1))).sum(axis=1)/np.log(max(p.shape[1],2))
    pair,pair_sum=distribution(pair_names);source,_=distribution(from_names);target,_=distribution(to_names);types,type_sum=distribution(type_names)
    middle=(source+target)/2
    js=.5*((source*(np.log(np.clip(source,1e-15,1))-np.log(np.clip(middle,1e-15,1)))).sum(1)+(target*(np.log(np.clip(target,1e-15,1))-np.log(np.clip(middle,1e-15,1)))).sum(1))/np.log(2)
    diagonal=[i for i,n in enumerate(pair_names) if n.split('|')[1].split('>')[0]==n.split('|')[1].split('>')[1]]
    columns={
        'composition|active_aa_pair_count':(pair>0).sum(1),
        'composition|aa_pair_entropy':entropy(pair),
        'composition|aa_pair_concentration':np.square(pair).sum(1),
        'composition|dominant_aa_pair_share':pair.max(1),
        'composition|source_aa_entropy':entropy(source),
        'composition|target_aa_entropy':entropy(target),
        'composition|source_target_js_divergence':np.maximum(js,0),
        'composition|synonymous_aa_pair_share':pair[:,diagonal].sum(1),
        'composition|no_simple_aa_information':pair_sum==0,
        'composition|type_mix_entropy':entropy(types),
        'composition|type_membership_per_mutated_gene':type_sum,
        'composition|no_type_information':type_sum==0,
    }
    values=np.column_stack(list(columns.values())).astype(np.float32)
    if not np.isfinite(values).all():raise ValueError('Nonfinite composition features')
    return values,list(columns)
