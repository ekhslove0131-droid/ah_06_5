from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


# ============================================================
# exp_51 | TRAIN-only Multi-Mutation Parser Screen
#
# 목적:
# 기존 NLP에는 남아 있던 multi-mutation 정보를
# XGBoost의 mutation type / position feature에도
# token 단위로 전달했을 때 성능이 개선되는지 확인한다.
#
# 변경하지 않는 것:
# - binary gene
# - mutation_count (= mutated gene/cell count)
# - signature
# - NLP
# - XGBoost hyperparameters
# - CV split
#
# 변경하는 것:
# - baseline: cell 전체를 mutation 1개처럼 parsing
# - candidate: cell을 whitespace token으로 나누고
#              각 token별 type + position parsing
#
# TEST는 절대 읽지 않는다.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "train.csv"

SEED = 7
N_SPLITS = 5
HASH_SIZE = 2 ** 15

start_time = time.time()

print("=" * 80)
print("exp_51 | TRAIN-only Multi-Mutation Parser Screen")
print("=" * 80)
print("TEST FILE: NOT READ")
print("Seed :", SEED)
print("Folds:", N_SPLITS)


# ============================================================
# 1. TRAIN
# ============================================================

train = pd.read_csv(TRAIN_PATH).reset_index(drop=True)

genes = [
    c for c in train.columns
    if c not in ["ID", "SUBCLASS"]
]

le = LabelEncoder()
y = le.fit_transform(train["SUBCLASS"])
n_classes = len(le.classes_)

print("\n환자 수 :", len(train))
print("유전자 수:", len(genes))
print("암종 수  :", n_classes)


# ============================================================
# 2. Binary + 기존 mutation_count
# ============================================================

binary_np = (
    train[genes].notna()
    & train[genes].ne("WT")
).to_numpy(dtype=np.uint8)

X_binary = sparse.csr_matrix(
    binary_np,
    dtype=np.float32
)

# 중요:
# 이 값은 mutation event 수가 아니라
# mutated gene/cell 수라는 기존 의미를 그대로 유지한다.
mutation_count = (
    binary_np
    .sum(axis=1)
    .astype(np.float32)
    .reshape(-1, 1)
)


# ============================================================
# 3. Parser helpers
# ============================================================

def classify_mutation_token(token):
    """
    기존 exp_34a와 동일한 규칙.
    이번 exp_51에서는 규칙 자체를 개선하지 않고,
    단지 cell 전체가 아니라 whitespace token마다 적용한다.

    index:
    0 = non-synonymous substitution
    1 = synonymous substitution
    2 = frameshift
    3 = stop
    4 = other
    """

    if "fs" in token.lower():
        return 2

    if "*" in token:
        return 3

    aa = re.search(
        r"([A-Z])(\d+)([A-Z])",
        token
    )

    if aa:
        before, _, after = aa.groups()

        if before == after:
            return 1

        return 0

    return 4


def first_position(token):
    """
    기존 position 규칙 유지:
    token 내부의 첫 숫자만 사용.
    복잡한 indel parser 개선은 이번 실험 범위가 아님.
    """

    pos = re.search(
        r"(\d+)",
        token
    )

    if pos:
        return int(pos.group(1))

    return None


# ============================================================
# 4. Baseline parser vs candidate parser
# ============================================================

baseline_type_counts = np.zeros(
    (len(train), 5),
    dtype=np.float32
)

candidate_type_counts = np.zeros(
    (len(train), 5),
    dtype=np.float32
)

baseline_position_tokens = [
    [] for _ in range(len(train))
]

candidate_position_tokens = [
    [] for _ in range(len(train))
]

document_tokens = [
    [] for _ in range(len(train))
]

mutations = (
    train[genes]
    .stack()
    .dropna()
)

mutations = mutations[
    mutations.ne("WT")
].astype(str)

baseline_position_found = 0
candidate_position_found = 0
candidate_token_count = 0
multi_token_cells = 0
max_tokens_per_cell = 0


for (row_idx, gene), mutation in mutations.items():

    # --------------------------------------------------------
    # A. Baseline parser
    # cell 전체를 하나의 mutation처럼 처리
    # --------------------------------------------------------

    baseline_type_idx = (
        classify_mutation_token(
            mutation
        )
    )

    baseline_type_counts[
        row_idx,
        baseline_type_idx
    ] += 1

    baseline_pos = (
        first_position(
            mutation
        )
    )

    if baseline_pos is not None:

        baseline_position_tokens[
            row_idx
        ].append(
            f"{gene}_POS{baseline_pos}"
        )

        baseline_position_found += 1


    # --------------------------------------------------------
    # B. Candidate parser
    # whitespace token별 처리
    # --------------------------------------------------------

    tokens = mutation.split()

    candidate_token_count += len(tokens)

    if len(tokens) >= 2:
        multi_token_cells += 1

    max_tokens_per_cell = max(
        max_tokens_per_cell,
        len(tokens)
    )

    for token in tokens:

        candidate_type_idx = (
            classify_mutation_token(
                token
            )
        )

        candidate_type_counts[
            row_idx,
            candidate_type_idx
        ] += 1

        candidate_pos = (
            first_position(
                token
            )
        )

        if candidate_pos is not None:

            candidate_position_tokens[
                row_idx
            ].append(
                f"{gene}_POS{candidate_pos}"
            )

            candidate_position_found += 1


    # --------------------------------------------------------
    # C. NLP
    # 기존 전체 문자열 그대로 유지
    # --------------------------------------------------------

    document_tokens[
        row_idx
    ].append(
        f"{gene}_{mutation}"
    )


documents = [
    " ".join(tokens)
    for tokens in document_tokens
]


# ============================================================
# 5. Parser diagnostics
# ============================================================

baseline_type_total = int(
    baseline_type_counts.sum()
)

candidate_type_total = int(
    candidate_type_counts.sum()
)

print("\n=== Parser diagnostics ===")

print(
    "non-WT cell 수:",
    len(mutations)
)

print(
    "multi-token cell 수:",
    multi_token_cells
)

print(
    "multi-token 비율:",
    round(
        multi_token_cells
        / len(mutations)
        * 100,
        4
    ),
    "%"
)

print(
    "전체 candidate mutation token:",
    candidate_token_count
)

print(
    "추가 token 수:",
    candidate_token_count
    - len(mutations)
)

print(
    "cell당 최대 token 수:",
    max_tokens_per_cell
)

print(
    "Baseline type count 합:",
    baseline_type_total
)

print(
    "Candidate type count 합:",
    candidate_type_total
)

print(
    "Candidate type 합 == token 수:",
    candidate_type_total
    == candidate_token_count
)

print(
    "Baseline 위치 token:",
    baseline_position_found
)

print(
    "Candidate 위치 token:",
    candidate_position_found
)


# ============================================================
# 6. Position hash + base matrices
# ============================================================

hasher = FeatureHasher(
    n_features=HASH_SIZE,
    input_type="string",
    alternate_sign=False
)

X_position_baseline = (
    hasher.transform(
        baseline_position_tokens
    )
)

X_position_candidate = (
    hasher.transform(
        candidate_position_tokens
    )
)

X_base_baseline = sparse.hstack(
    [
        X_binary,
        sparse.csr_matrix(
            mutation_count
        ),
        sparse.csr_matrix(
            baseline_type_counts
        ),
        X_position_baseline,
    ],
    format="csr"
)

X_base_candidate = sparse.hstack(
    [
        X_binary,
        sparse.csr_matrix(
            mutation_count
        ),
        sparse.csr_matrix(
            candidate_type_counts
        ),
        X_position_candidate,
    ],
    format="csr"
)

print(
    "\nBaseline feature 수 :",
    X_base_baseline.shape[1]
)

print(
    "Candidate feature 수:",
    X_base_candidate.shape[1]
)


# ============================================================
# 7. Signature
# ============================================================

def make_centroids(
    X_bin,
    labels,
    reference_idx,
    n_classes
):

    centroids = np.zeros(
        (
            n_classes,
            X_bin.shape[1]
        ),
        dtype=np.float32
    )

    for c in range(n_classes):

        class_rows = reference_idx[
            labels[
                reference_idx
            ] == c
        ]

        if len(class_rows) > 0:

            centroids[c] = (
                np.asarray(
                    X_bin[
                        class_rows
                    ].mean(axis=0)
                )
                .ravel()
                .astype(np.float32)
            )

    return centroids


def cosine_signature(
    X_rows,
    centroids
):

    dot = np.asarray(
        X_rows @ centroids.T,
        dtype=np.float32
    )

    patient_norm = np.sqrt(
        np.asarray(
            X_rows
            .multiply(X_rows)
            .sum(axis=1)
        ).ravel()
    ).astype(np.float32)

    centroid_norm = np.linalg.norm(
        centroids,
        axis=1
    ).astype(np.float32)

    denom = (
        patient_norm[:, None]
        * centroid_norm[None, :]
    )

    return np.divide(
        dot,
        denom,
        out=np.zeros_like(dot),
        where=denom > 0
    ).astype(np.float32)


def make_crossfit_signature(
    train_idx,
    valid_idx,
    seed
):

    signature_train = np.zeros(
        (
            len(train_idx),
            n_classes
        ),
        dtype=np.float32
    )

    y_outer = y[
        train_idx
    ]

    inner_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for (
        inner_train_pos,
        inner_valid_pos
    ) in inner_cv.split(
        np.zeros(len(train_idx)),
        y_outer
    ):

        reference_idx = (
            train_idx[
                inner_train_pos
            ]
        )

        inner_valid_global = (
            train_idx[
                inner_valid_pos
            ]
        )

        centroids_inner = (
            make_centroids(
                X_binary,
                y,
                reference_idx,
                n_classes
            )
        )

        signature_train[
            inner_valid_pos
        ] = cosine_signature(
            X_binary[
                inner_valid_global
            ],
            centroids_inner
        )

    centroids_outer = (
        make_centroids(
            X_binary,
            y,
            train_idx,
            n_classes
        )
    )

    signature_valid = (
        cosine_signature(
            X_binary[
                valid_idx
            ],
            centroids_outer
        )
    )

    return (
        signature_train,
        signature_valid
    )


# ============================================================
# 8. Models
# ============================================================

def make_xgb(seed):

    return XGBClassifier(
        n_estimators=200,
        learning_rate=0.10,
        max_depth=3,
        random_state=seed,
        n_jobs=6,
        tree_method="hist",
    )


def make_vectorizer():

    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=120000,
        sublinear_tf=True,
        norm="l2",
        lowercase=False,
        dtype=np.float32,
    )


def make_nlp(seed):

    return LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=seed,
        max_iter=5000,
    )


# ============================================================
# 9. OOF containers
# ============================================================

baseline_xgb_oof = np.zeros(
    len(train),
    dtype=np.int32
)

candidate_xgb_oof = np.zeros(
    len(train),
    dtype=np.int32
)

nlp_oof = np.zeros(
    len(train),
    dtype=np.int32
)

baseline_ensemble_oof = np.zeros(
    len(train),
    dtype=np.int32
)

candidate_ensemble_oof = np.zeros(
    len(train),
    dtype=np.int32
)


baseline_xgb_prob_oof = np.zeros(
    (
        len(train),
        n_classes
    ),
    dtype=np.float32
)

candidate_xgb_prob_oof = np.zeros(
    (
        len(train),
        n_classes
    ),
    dtype=np.float32
)

nlp_prob_oof = np.zeros(
    (
        len(train),
        n_classes
    ),
    dtype=np.float32
)

baseline_ensemble_prob_oof = np.zeros(
    (
        len(train),
        n_classes
    ),
    dtype=np.float32
)

candidate_ensemble_prob_oof = np.zeros(
    (
        len(train),
        n_classes
    ),
    dtype=np.float32
)


fold_rows = []
prediction_rows = []


# ============================================================
# 10. Outer CV
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=SEED
)


for fold, (
    train_idx,
    valid_idx
) in enumerate(
    outer_cv.split(
        np.zeros(len(train)),
        y
    ),
    start=1
):

    fold_start = time.time()

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"Seed {SEED} | Fold {fold}/{N_SPLITS}"
    )

    print(
        "-" * 80
    )


    # --------------------------------------------------------
    # A. Signature
    # binary가 동일하므로 baseline/candidate가 완전히 공유
    # --------------------------------------------------------

    (
        signature_train,
        signature_valid
    ) = make_crossfit_signature(
        train_idx,
        valid_idx,
        SEED
    )


    # --------------------------------------------------------
    # B. Baseline XGB
    # --------------------------------------------------------

    X_baseline_train = sparse.hstack(
        [
            X_base_baseline[
                train_idx
            ],
            sparse.csr_matrix(
                signature_train
            )
        ],
        format="csr"
    )

    X_baseline_valid = sparse.hstack(
        [
            X_base_baseline[
                valid_idx
            ],
            sparse.csr_matrix(
                signature_valid
            )
        ],
        format="csr"
    )

    baseline_model = (
        make_xgb(
            SEED
        )
    )

    baseline_model.fit(
        X_baseline_train,
        y[
            train_idx
        ]
    )

    baseline_xgb_proba = (
        baseline_model
        .predict_proba(
            X_baseline_valid
        )
    )

    baseline_xgb_pred = np.argmax(
        baseline_xgb_proba,
        axis=1
    ).astype(np.int32)


    # --------------------------------------------------------
    # C. Candidate XGB
    # --------------------------------------------------------

    X_candidate_train = sparse.hstack(
        [
            X_base_candidate[
                train_idx
            ],
            sparse.csr_matrix(
                signature_train
            )
        ],
        format="csr"
    )

    X_candidate_valid = sparse.hstack(
        [
            X_base_candidate[
                valid_idx
            ],
            sparse.csr_matrix(
                signature_valid
            )
        ],
        format="csr"
    )

    candidate_model = (
        make_xgb(
            SEED
        )
    )

    candidate_model.fit(
        X_candidate_train,
        y[
            train_idx
        ]
    )

    candidate_xgb_proba = (
        candidate_model
        .predict_proba(
            X_candidate_valid
        )
    )

    candidate_xgb_pred = np.argmax(
        candidate_xgb_proba,
        axis=1
    ).astype(np.int32)


    # --------------------------------------------------------
    # D. NLP
    # fold당 딱 한 번만 학습
    # 양쪽 앙상블이 같은 확률을 공유한다.
    # --------------------------------------------------------

    train_docs = [
        documents[i]
        for i in train_idx
    ]

    valid_docs = [
        documents[i]
        for i in valid_idx
    ]

    vectorizer = (
        make_vectorizer()
    )

    X_nlp_train = (
        vectorizer
        .fit_transform(
            train_docs
        )
    )

    X_nlp_valid = (
        vectorizer
        .transform(
            valid_docs
        )
    )

    nlp_model = (
        make_nlp(
            SEED
        )
    )

    nlp_calibrated = (
        CalibratedClassifierCV(
            nlp_model,
            method="sigmoid",
            cv=3
        )
    )

    nlp_calibrated.fit(
        X_nlp_train,
        y[
            train_idx
        ]
    )

    nlp_proba = (
        nlp_calibrated
        .predict_proba(
            X_nlp_valid
        )
    )

    nlp_pred = np.argmax(
        nlp_proba,
        axis=1
    ).astype(np.int32)


    # --------------------------------------------------------
    # E. 동일 NLP를 사용한 80:20 ensemble
    # --------------------------------------------------------

    baseline_ensemble_proba = (
        0.80
        * baseline_xgb_proba
        + 0.20
        * nlp_proba
    )

    candidate_ensemble_proba = (
        0.80
        * candidate_xgb_proba
        + 0.20
        * nlp_proba
    )

    baseline_ensemble_pred = np.argmax(
        baseline_ensemble_proba,
        axis=1
    ).astype(np.int32)

    candidate_ensemble_pred = np.argmax(
        candidate_ensemble_proba,
        axis=1
    ).astype(np.int32)


    # --------------------------------------------------------
    # F. OOF 저장
    # --------------------------------------------------------

    baseline_xgb_oof[
        valid_idx
    ] = baseline_xgb_pred

    candidate_xgb_oof[
        valid_idx
    ] = candidate_xgb_pred

    nlp_oof[
        valid_idx
    ] = nlp_pred

    baseline_ensemble_oof[
        valid_idx
    ] = baseline_ensemble_pred

    candidate_ensemble_oof[
        valid_idx
    ] = candidate_ensemble_pred


    baseline_xgb_prob_oof[
        valid_idx
    ] = baseline_xgb_proba

    candidate_xgb_prob_oof[
        valid_idx
    ] = candidate_xgb_proba

    nlp_prob_oof[
        valid_idx
    ] = nlp_proba

    baseline_ensemble_prob_oof[
        valid_idx
    ] = baseline_ensemble_proba

    candidate_ensemble_prob_oof[
        valid_idx
    ] = candidate_ensemble_proba


    # --------------------------------------------------------
    # G. Fold metrics
    # --------------------------------------------------------

    truth = y[
        valid_idx
    ]

    baseline_xgb_f1 = f1_score(
        truth,
        baseline_xgb_pred,
        average="macro"
    )

    candidate_xgb_f1 = f1_score(
        truth,
        candidate_xgb_pred,
        average="macro"
    )

    baseline_ensemble_f1 = f1_score(
        truth,
        baseline_ensemble_pred,
        average="macro"
    )

    candidate_ensemble_f1 = f1_score(
        truth,
        candidate_ensemble_pred,
        average="macro"
    )

    xgb_diff = (
        candidate_xgb_f1
        - baseline_xgb_f1
    )

    ensemble_diff = (
        candidate_ensemble_f1
        - baseline_ensemble_f1
    )

    elapsed = (
        time.time()
        - fold_start
    )

    fold_rows.append(
        {
            "seed":
                SEED,
            "fold":
                fold,
            "baseline_xgb_f1":
                baseline_xgb_f1,
            "candidate_xgb_f1":
                candidate_xgb_f1,
            "xgb_diff":
                xgb_diff,
            "baseline_80_20_f1":
                baseline_ensemble_f1,
            "candidate_80_20_f1":
                candidate_ensemble_f1,
            "ensemble_diff":
                ensemble_diff,
            "seconds":
                elapsed,
        }
    )


    for pos, row_idx in enumerate(
        valid_idx
    ):

        prediction_rows.append(
            {
                "seed":
                    SEED,
                "fold":
                    fold,
                "row_index":
                    int(row_idx),
                "ID":
                    train.loc[
                        row_idx,
                        "ID"
                    ],
                "true_subclass":
                    le.inverse_transform(
                        [truth[pos]]
                    )[0],
                "baseline_xgb_pred":
                    le.inverse_transform(
                        [baseline_xgb_pred[pos]]
                    )[0],
                "candidate_xgb_pred":
                    le.inverse_transform(
                        [candidate_xgb_pred[pos]]
                    )[0],
                "nlp_pred":
                    le.inverse_transform(
                        [nlp_pred[pos]]
                    )[0],
                "baseline_ensemble_pred":
                    le.inverse_transform(
                        [baseline_ensemble_pred[pos]]
                    )[0],
                "candidate_ensemble_pred":
                    le.inverse_transform(
                        [candidate_ensemble_pred[pos]]
                    )[0],
            }
        )


    print(
        f"Baseline XGB      : "
        f"{baseline_xgb_f1:.6f}"
    )

    print(
        f"Candidate XGB     : "
        f"{candidate_xgb_f1:.6f}"
    )

    print(
        f"XGB 차이          : "
        f"{xgb_diff:+.6f}"
    )

    print(
        f"Baseline 80:20    : "
        f"{baseline_ensemble_f1:.6f}"
    )

    print(
        f"Candidate 80:20   : "
        f"{candidate_ensemble_f1:.6f}"
    )

    print(
        f"앙상블 차이       : "
        f"{ensemble_diff:+.6f}"
    )

    print(
        f"Fold time         : "
        f"{elapsed:.1f}s"
    )


# ============================================================
# 11. Overall OOF metrics
# ============================================================

baseline_xgb_f1 = f1_score(
    y,
    baseline_xgb_oof,
    average="macro"
)

candidate_xgb_f1 = f1_score(
    y,
    candidate_xgb_oof,
    average="macro"
)

baseline_ensemble_f1 = f1_score(
    y,
    baseline_ensemble_oof,
    average="macro"
)

candidate_ensemble_f1 = f1_score(
    y,
    candidate_ensemble_oof,
    average="macro"
)

xgb_diff = (
    candidate_xgb_f1
    - baseline_xgb_f1
)

ensemble_diff = (
    candidate_ensemble_f1
    - baseline_ensemble_f1
)

fold_df = pd.DataFrame(
    fold_rows
)

prediction_df = pd.DataFrame(
    prediction_rows
)

improved_folds = int(
    (
        fold_df[
            "ensemble_diff"
        ] > 0
    ).sum()
)


# ============================================================
# 12. Baseline reproduction audit
# ============================================================

old_prediction_path = (
    BASE_DIR
    / "exp_34a_base_newseeds_predictions.csv"
)

baseline_reproduction_rate = np.nan
baseline_mismatch_count = -1

if old_prediction_path.exists():

    old = pd.read_csv(
        old_prediction_path
    )

    old = old[
        old["seed"] == SEED
    ].copy()

    old = old.sort_values(
        "row_index"
    )

    if (
        len(old) == len(train)
        and old[
            "row_index"
        ].nunique() == len(train)
    ):

        old_pred = le.transform(
            old[
                "ensemble_pred"
            ]
        )

        baseline_reproduction_rate = float(
            np.mean(
                old_pred
                == baseline_ensemble_oof
            )
        )

        baseline_mismatch_count = int(
            np.sum(
                old_pred
                != baseline_ensemble_oof
            )
        )


baseline_exact = bool(
    baseline_reproduction_rate == 1.0
)


# ============================================================
# 13. Rescue / damage
# ============================================================

baseline_correct = (
    baseline_ensemble_oof
    == y
)

candidate_correct = (
    candidate_ensemble_oof
    == y
)

candidate_rescue = int(
    np.sum(
        ~baseline_correct
        & candidate_correct
    )
)

candidate_damage = int(
    np.sum(
        baseline_correct
        & ~candidate_correct
    )
)

net_correct_gain = (
    candidate_rescue
    - candidate_damage
)


# ============================================================
# 14. Classwise metrics
# ============================================================

labels = np.arange(
    n_classes
)

baseline_class_f1 = f1_score(
    y,
    baseline_ensemble_oof,
    labels=labels,
    average=None,
    zero_division=0
)

candidate_class_f1 = f1_score(
    y,
    candidate_ensemble_oof,
    labels=labels,
    average=None,
    zero_division=0
)

classwise_rows = []

for class_idx, class_name in enumerate(
    le.classes_
):

    class_mask = (
        y == class_idx
    )

    class_baseline_correct = (
        baseline_ensemble_oof[
            class_mask
        ] == class_idx
    )

    class_candidate_correct = (
        candidate_ensemble_oof[
            class_mask
        ] == class_idx
    )

    rescue = int(
        np.sum(
            ~class_baseline_correct
            & class_candidate_correct
        )
    )

    damage = int(
        np.sum(
            class_baseline_correct
            & ~class_candidate_correct
        )
    )

    classwise_rows.append(
        {
            "subclass":
                class_name,
            "support":
                int(
                    class_mask.sum()
                ),
            "baseline_f1":
                baseline_class_f1[
                    class_idx
                ],
            "candidate_f1":
                candidate_class_f1[
                    class_idx
                ],
            "f1_diff":
                candidate_class_f1[
                    class_idx
                ]
                - baseline_class_f1[
                    class_idx
                ],
            "rescue":
                rescue,
            "damage":
                damage,
            "net_correct_gain":
                rescue - damage,
        }
    )

classwise_df = pd.DataFrame(
    classwise_rows
)


# ============================================================
# 15. 26-class OOF probabilities
# ============================================================

prob_df = pd.DataFrame(
    {
        "row_index":
            np.arange(
                len(train)
            ),
        "ID":
            train["ID"],
        "true_subclass":
            train["SUBCLASS"],
    }
)

for class_idx, class_name in enumerate(
    le.classes_
):

    prob_df[
        f"baseline_xgb_prob_{class_name}"
    ] = baseline_xgb_prob_oof[
        :,
        class_idx
    ]

    prob_df[
        f"candidate_xgb_prob_{class_name}"
    ] = candidate_xgb_prob_oof[
        :,
        class_idx
    ]

    prob_df[
        f"nlp_prob_{class_name}"
    ] = nlp_prob_oof[
        :,
        class_idx
    ]

    prob_df[
        f"baseline_ensemble_prob_{class_name}"
    ] = baseline_ensemble_prob_oof[
        :,
        class_idx
    ]

    prob_df[
        f"candidate_ensemble_prob_{class_name}"
    ] = candidate_ensemble_prob_oof[
        :,
        class_idx
    ]


# ============================================================
# 16. PASS / STOP
# ============================================================

screen_pass = bool(
    baseline_exact
    and ensemble_diff >= 0.003
    and improved_folds >= 4
)

seed_df = pd.DataFrame(
    [
        {
            "seed":
                SEED,
            "baseline_xgb_oof_f1":
                baseline_xgb_f1,
            "candidate_xgb_oof_f1":
                candidate_xgb_f1,
            "xgb_diff":
                xgb_diff,
            "baseline_80_20_oof_f1":
                baseline_ensemble_f1,
            "candidate_80_20_oof_f1":
                candidate_ensemble_f1,
            "ensemble_diff":
                ensemble_diff,
            "improved_folds":
                improved_folds,
            "candidate_rescue":
                candidate_rescue,
            "candidate_damage":
                candidate_damage,
            "net_correct_gain":
                net_correct_gain,
            "baseline_reproduction_rate":
                baseline_reproduction_rate,
            "baseline_mismatch_count":
                baseline_mismatch_count,
            "screen_pass":
                screen_pass,
        }
    ]
)


# ============================================================
# 17. Save — exp_51 전용 파일명
# ============================================================

seed_df.to_csv(
    BASE_DIR
    / "exp_51_multi_mutation_seed_results.csv",
    index=False
)

fold_df.to_csv(
    BASE_DIR
    / "exp_51_multi_mutation_fold_results.csv",
    index=False
)

prediction_df.to_csv(
    BASE_DIR
    / "exp_51_multi_mutation_predictions.csv",
    index=False
)

classwise_df.to_csv(
    BASE_DIR
    / "exp_51_multi_mutation_classwise_results.csv",
    index=False
)

prob_df.to_csv(
    BASE_DIR
    / "exp_51_multi_mutation_oof_probabilities.csv",
    index=False
)


# ============================================================
# 18. Final report
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "exp_51 MULTI-MUTATION PARSER SCREEN SUMMARY"
)

print(
    "=" * 80
)

print(
    f"Baseline XGB OOF     : "
    f"{baseline_xgb_f1:.6f}"
)

print(
    f"Candidate XGB OOF    : "
    f"{candidate_xgb_f1:.6f}"
)

print(
    f"XGB 차이             : "
    f"{xgb_diff:+.6f}"
)

print()

print(
    f"Baseline 80:20 OOF   : "
    f"{baseline_ensemble_f1:.6f}"
)

print(
    f"Candidate 80:20 OOF  : "
    f"{candidate_ensemble_f1:.6f}"
)

print(
    f"주 지표 차이         : "
    f"{ensemble_diff:+.6f}"
)

print(
    f"개선 Fold            : "
    f"{improved_folds}/{N_SPLITS}"
)

print()

print(
    f"Candidate rescue     : "
    f"{candidate_rescue}"
)

print(
    f"Candidate damage     : "
    f"{candidate_damage}"
)

print(
    f"정답 순증가          : "
    f"{net_correct_gain:+d}"
)

print()

print(
    "기존 exp_34a seed7 재현율:",
    baseline_reproduction_rate
)

print(
    "기존 baseline 불일치:",
    baseline_mismatch_count
)

print(
    "Baseline exact reproduction:",
    baseline_exact
)

print()

print(
    "합격 기준:"
)

print(
    "1) 기존 baseline 100% 재현"
)

print(
    "2) 80:20 Macro F1 +0.003 이상"
)

print(
    "3) 4/5 folds 이상 개선"
)

print(
    "\nSCREEN RESULT:",
    (
        "PASS ✅"
        if screen_pass
        else "STOP / HOLD ❌"
    )
)

if not baseline_exact:

    print(
        "\n⚠️ baseline이 기존 exp_34a seed7과 "
        "완전히 같지 않으므로 parser 효과 해석을 중단해야 합니다."
    )

print(
    "\nTEST FILE: NEVER READ"
)

print(
    "총 걸린 시간:",
    round(
        time.time()
        - start_time,
        1
    ),
    "초"
)

print(
    "=" * 80
)
