"""MCMP FINAL_v2 feature semantics, extracted without legacy I/O or model runners.
Functions come from the verified MCMP embedded PFIX/legacy source. New Encoder
persists the same train-only DictVectorizer vocabulary as a portable JSON mapping.
"""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

SIMPLE_SUB_RE = re.compile(r"^([A-Z])(\d+)([A-Z*X])$", re.I)


NUM_RE = re.compile(r"\d+")


def _inc(d, key, value=1.0):
    d[key] = d.get(key, 0.0) + float(value)


def _legacy_classify_token(token: str):
    t = token.upper()
    m = SIMPLE_SUB_RE.fullmatch(t)
    if "FS" in t:
        kind = "frameshift"
    elif "DELINS" in t:
        kind = "delins"
    elif "DEL" in t:
        kind = "deletion"
    elif "INS" in t:
        kind = "insertion"
    elif m:
        src, _, dst = m.groups()
        if dst in ("*","X"):
            kind = "stop_gained"
        elif src in ("*","X"):
            kind = "stop_lost"
        elif src == dst:
            kind = "synonymous"
        else:
            kind = "missense"
    else:
        kind = "complex_unparsed"
    nums = NUM_RE.findall(t)
    if len(nums) == 1:
        site = nums[0]
    elif len(nums) >= 2:
        site = nums[0] + "-" + nums[1]
    else:
        site = None
    aa_change = None
    aa_from = aa_to = None
    if m:
        aa_from, _, aa_to = m.groups()
        aa_change = f"{aa_from}>{aa_to}"
    return kind, site, aa_change, aa_from, aa_to


def build_feature_rows(train: pd.DataFrame, gene_cols: list[str]):
    rows = [dict() for _ in range(len(train))]
    stats = Counter()
    type_counts = Counter()
    for gene in gene_cols:
        series = train[gene]
        missing_mask = series.isna() | series.astype(str).str.strip().eq("")
        for idx in np.flatnonzero(missing_mask.to_numpy()):
            rows[idx][f"gene|{gene}|missing"] = 1.0
            stats["missing_cells"] += 1
        nonwt_mask = (~missing_mask) & series.astype(str).str.upper().ne("WT")
        for idx in np.flatnonzero(nonwt_mask.to_numpy()):
            raw = series.iloc[idx]
            rows[idx][f"gene|{gene}|nonWT"] = 1.0
            stats["mutation_cells"] += 1
            tokens = split_tokens(raw)
            if not tokens:
                rows[idx][f"type|{gene}|complex_unparsed"] = 1.0
                type_counts["complex_unparsed"] += 1
                continue
            for token in tokens:
                stats["tokens"] += 1
                kind, site, change, aa_from, aa_to = classify_token(token)
                rows[idx][f"type|{gene}|{kind}"] = 1.0
                type_counts[kind] += 1
                rows[idx][f"exact|{gene}|{token}"] = 1.0
                if site is not None:
                    rows[idx][f"site|{gene}|{site}"] = 1.0
                if change is not None:
                    rows[idx][f"gene_change|{gene}|{change}"] = 1.0
                    _inc(rows[idx], f"aa|{change}")
                    _inc(rows[idx], f"aa_from|{aa_from}")
                    _inc(rows[idx], f"aa_to|{aa_to}")
                    if aa_from != aa_to:
                        stats["simple_nonsyn_substitutions"] += 1
                if kind == "complex_unparsed":
                    stats["unparsed_tokens"] += 1
    for d in rows:
        mut_count = sum(1 for k in d if k.startswith("gene|") and k.endswith("|nonWT"))
        missing_count = sum(1 for k in d if k.startswith("gene|") and k.endswith("|missing"))
        d["global|mutation_gene_count"] = float(mut_count)
        d["global|missing_gene_count"] = float(missing_count)
        d["global|observed_gene_count"] = float(len(gene_cols) - missing_count)
    return rows, {"counts": dict(stats), "type_counts": dict(type_counts)}


def block_of(feature: str) -> str:
    return feature.split("|", 1)[0]


def _filtered_row(d, blocks, fine_allowed):
    out = {}
    for k,v in d.items():
        b = block_of(k)
        if b not in blocks:
            continue
        if b in ("gene_change","site","exact") and k not in fine_allowed:
            continue
        out[k] = v
    return out

SIMPLE = re.compile(r'^([A-Z])(\d+)([A-Z*X])$', re.I)


FS = re.compile(r'^[A-Z](\d+)(?:[A-Z*])?FS(?:(?:\*|X|TER)?(?:\d+|\?))?$', re.I)


INDEL = re.compile(r'^[A-Z](\d+)(?:_[A-Z](\d+))?(DELINS|DEL|INS)(?:[A-Z*]*)$', re.I)


TOKEN_START = re.compile(r'^[A-Z*]{1,3}\d+', re.I)


def split_tokens(value):
    """Split independent compact annotations only; preserve unresolved compound expressions.
    No attempt is made to decompose HGVS allelic/bracket relationships or natural language.
    """
    if pd.isna(value): return []
    text=str(value).strip()
    if not text or text.upper()=='WT': return []
    # Group/allele notation is not a list of independent substitutions.
    if any(ch in text for ch in '[]()'):
        return [text.upper()]
    result=[]
    for chunk in re.split(r'[;,]+',text):
        chunk=chunk.strip()
        if not chunk: continue
        parts=re.split(r'\s+',chunk)
        if len(parts)>1 and all(TOKEN_START.match(part) for part in parts):
            result.extend(part.upper() for part in parts)
        else:
            result.append(chunk.upper())
    return result


def safe_site(token):
    """Project compact protein annotations, not a general HGVS/reference validator.
    fs termination-distance numbers are never treated as coordinate endpoints.
    """
    token=token.upper()
    sub=SIMPLE.fullmatch(token)
    if sub:
        p=int(sub.group(2)); return (str(p),'substitution') if p>0 else (None,'invalid_position')
    fs=FS.fullmatch(token)
    if fs:
        p=int(fs.group(1)); return (str(p),'frameshift_start') if p>0 else (None,'invalid_position')
    indel=INDEL.fullmatch(token)
    if indel:
        start=int(indel.group(1)); end=int(indel.group(2)) if indel.group(2) else None
        if start<=0 or (end is not None and end<start): return None,'invalid_range'
        if indel.group(3).upper()=='INS' and end is None: return None,'ambiguous_insertion'
        return (f'{start}-{end}' if end is not None else str(start)), 'indel_coordinates'
    return None,'unresolved_coordinates'


def classify_token(token):
    kind, _, change, aa_from, aa_to = _legacy_classify_token(token)
    site, _ = safe_site(token)
    return kind, site, change, aa_from, aa_to


class Encoder:
    def __init__(self, blocks, min_support):
        self.blocks = list(blocks)
        self.min_support = int(min_support)
        self.allowed = set()
        self.names = []
        self.index = {}

    def fit_transform(self, rows, groups):
        support = defaultdict(set)
        for row, group in zip(rows, groups):
            for name in row:
                if block_of(name) in {'gene_change','site','exact'} and block_of(name) in self.blocks:
                    support[name].add(str(group))
        self.allowed = {name for name, gs in support.items() if len(gs) >= self.min_support}
        vec = DictVectorizer(sparse=True, sort=True)
        x = vec.fit_transform([_filtered_row(r, set(self.blocks), self.allowed) for r in rows]).astype(np.float32).tocsr()
        self.names = vec.get_feature_names_out().tolist()
        self.index = {v:i for i,v in enumerate(self.names)}
        return x

    def transform(self, rows):
        data=[]; indices=[]; indptr=[0]
        for row in rows:
            pairs=sorted((self.index[key], float(value)) for key,value in row.items() if key in self.index)
            indices.extend(i for i,v in pairs); data.extend(v for i,v in pairs);indptr.append(len(data))
        return sparse.csr_matrix((np.array(data,np.float32),np.array(indices,np.int32),np.array(indptr,np.int32)),shape=(len(rows),len(self.names)))

    def to_dict(self):
        return {'kind':'MCMP_corrected_detail','blocks':self.blocks,'min_support':self.min_support,
                'names':self.names,'fine_allowed':sorted(self.allowed)}

    @classmethod
    def from_dict(cls,d):
        obj=cls(d['blocks'],d['min_support']);obj.names=list(d['names']);obj.allowed=set(d['fine_allowed'])
        obj.index={n:i for i,n in enumerate(obj.names)}
        return obj
