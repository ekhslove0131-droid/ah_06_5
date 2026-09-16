"""Resolve a ledger alias to the exact computation originally scored."""

import math


def _single_source_signature(recipe):
    sources, weights = recipe['source_recipe_ids'], recipe['weights']
    if len(sources) != len(weights) or not weights:
        raise ValueError('invalid alias weights')
    if any(not math.isfinite(float(w)) or w < 0 for w in weights) or abs(sum(weights) - 1) > 1e-12:
        raise ValueError('invalid alias weights')
    active = tuple((source, float(weight)) for source, weight in zip(sources, weights) if weight > 0)
    if len(active) != 1:
        raise ValueError('only single-source pooling aliases may be canonicalized')
    return recipe['kind'], recipe['round'], active, recipe['bias_strength']


def execution_row(row, index):
    current, seen = row, set()
    while current.get('alias_of') and current.get('effective_key'):
        if current['recipe_id'] in seen:
            raise ValueError('cyclic execution alias')
        seen.add(current['recipe_id'])
        reference = index.get(current['alias_of'])
        if reference is None:
            raise ValueError('execution alias target is missing')
        if reference.get('effective_key') != current['effective_key']:
            return current
        if (current['probability_sha256'] != reference['probability_sha256']
                or current['macro_f1'] != reference['macro_f1']
                or _single_source_signature(current['recipe']) != _single_source_signature(reference['recipe'])):
            raise ValueError('execution alias evidence differs')
        current = reference
    return current
