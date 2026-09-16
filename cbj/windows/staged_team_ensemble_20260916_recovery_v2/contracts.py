"""Frozen identities and schema constants for the team ensemble search."""

from __future__ import annotations


TRAIN_SHA256 = "92418b8441d058cfc68e939dd88725610750be4bc8edc51253cffc72fc4fc0ab"
FOLD_SHA256 = "e718a5c311da397199dd457c80ba7c5c51a89f05b4f17c8c12f93b08e790b0e8"
REFERENCE_OOF_SHA256 = "ab952812c5b779bbaee826bc9163579268cffbf50dd567fb2162c5600bec08e9"
ROWS = 6201
CLASS_COUNT = 26
GROUP_COUNT = 5636
CLASS_ORDER = [
    "ACC", "BLCA", "BRCA", "CESC", "COAD", "DLBC", "GBMLGG", "HNSC",
    "KIPAN", "KIRC", "LAML", "LGG", "LIHC", "LUAD", "LUSC", "OV",
    "PAAD", "PCPG", "PRAD", "SARC", "SKCM", "STES", "TGCT", "THCA",
    "THYM", "UCEC",
]
SPLIT_CONTRACT = {
    "id": "canonical_group5_seed43",
    "train_sha256": TRAIN_SHA256,
    "fold_sha256": FOLD_SHA256,
    "reference_oof_sha256": REFERENCE_OOF_SHA256,
    "rows": ROWS,
    "classes": CLASS_COUNT,
    "groups": GROUP_COUNT,
    "class_order": CLASS_ORDER,
}

