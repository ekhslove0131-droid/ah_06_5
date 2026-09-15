import re
import time
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier


print("🧬 exp_05 | BEST 80% + NLP 20% 앙상블")
start_time = time.time()


# ==================================================
# 1. 데이터
# ==================================================
train = pd.read_csv("train.csv").reset_index(drop=True)

genes = [
    c for c in train.columns
    if c not in ["ID", "SUBCLASS"]
]

le = LabelEncoder()
y = le.fit_transform(train["SUBCLASS"])

n_classes = len(le.classes_)
all_idx = np.arange(len(train))

print("환자 수:", len(train))
print("유전자 수:", len(genes))
print("암종 수:", n_classes)


# ==================================================
# 2. Binary
# ==================================================
binary_np = (
    train[genes].notna()
    & train[genes].ne("WT")
).to_numpy(dtype=np.uint8)

X_binary = sparse.csr_matrix(
    binary_np,
    dtype=np.float32
)

mutation_count = (
    binary_np.sum(axis=1)
    .astype(np.float32)
    .reshape(-1, 1)
)


# ==================================================
# 3. Mutation type + gene_position + NLP document
# ==================================================
type_counts = np.zeros(
    (len(train), 5),
    dtype=np.float32
)

position_tokens = [
    [] for _ in range(len(train))
]

document_tokens = [
    [] for _ in range(len(train))
]

mutations = train[genes].stack().dropna()
mutations = mutations[
    mutations.ne("WT")
].astype(str)

position_found = 0

for (row_idx, gene), mutation in mutations.items():

    # --------------------------
    # mutation type
    # --------------------------
    if "fs" in mutation.lower():
        type_counts[row_idx, 2] += 1

    elif "*" in mutation:
        type_counts[row_idx, 3] += 1

    else:
        aa = re.search(
            r"([A-Z])(\d+)([A-Z])",
            mutation
        )

        if aa:
            before, _, after = aa.groups()

            if before == after:
                type_counts[row_idx, 1] += 1
            else:
                type_counts[row_idx, 0] += 1

        else:
            type_counts[row_idx, 4] += 1

    # --------------------------
    # gene_position
    # --------------------------
    pos = re.search(r"(\d+)", mutation)

    if pos:
        position = int(pos.group(1))

        position_tokens[row_idx].append(
            f"{gene}_POS{position}"
        )

        position_found += 1

    # --------------------------
    # NLP mutation document
    # --------------------------
    document_tokens[row_idx].append(
        f"{gene}_{mutation}"
    )


documents = [
    " ".join(tokens)
    for tokens in document_tokens
]


print(
    "변이종류 합계 체크:",
    np.allclose(
        type_counts.sum(axis=1),
        mutation_count.ravel()
    )
)

print(
    "위치 추출:",
    position_found,
    "/",
    len(mutations)
)


# ==================================================
# 4. gene_position Hash
# ==================================================
HASH_SIZE = 2 ** 15

hasher = FeatureHasher(
    n_features=HASH_SIZE,
    input_type="string",
    alternate_sign=False
)

X_position = hasher.transform(
    position_tokens
)


# ==================================================
# 5. BEST 기본 피처
# ==================================================
X_base = sparse.hstack(
    [
        X_binary,
        sparse.csr_matrix(mutation_count),
        sparse.csr_matrix(type_counts),
        X_position,
    ],
    format="csr"
)

print("BEST 기본 피처 수:", X_base.shape[1])

# ==================================================
# 5-1. BEST07용 Multi-Mutation X_base
# exp32는 기존 X_base를 그대로 사용한다.
# mutation_count / binary / NLP는 기존과 동일.
# type + position만 whitespace token 단위로 계산.
# ==================================================
type_counts_multi = np.zeros(
    (len(train), 5),
    dtype=np.float32
)

position_tokens_multi = [
    [] for _ in range(len(train))
]

for (row_idx, gene), mutation in mutations.items():

    for mutation_token in str(mutation).split():

        if "fs" in mutation_token.lower():
            type_counts_multi[row_idx, 2] += 1

        elif "*" in mutation_token:
            type_counts_multi[row_idx, 3] += 1

        else:
            aa = re.search(
                r"([A-Z])(\d+)([A-Z])",
                mutation_token
            )

            if aa:
                before, _, after = aa.groups()

                if before == after:
                    type_counts_multi[row_idx, 1] += 1
                else:
                    type_counts_multi[row_idx, 0] += 1

            else:
                type_counts_multi[row_idx, 4] += 1

        pos = re.search(
            r"(\d+)",
            mutation_token
        )

        if pos:
            position = int(pos.group(1))

            position_tokens_multi[row_idx].append(
                f"{gene}_POS{position}"
            )

X_position_multi = hasher.transform(
    position_tokens_multi
)

X_base_multi = sparse.hstack(
    [
        X_binary,
        sparse.csr_matrix(mutation_count),
        sparse.csr_matrix(type_counts_multi),
        X_position_multi,
    ],
    format="csr"
)

print(
    "BEST07 Multi-parser 기본 피처 수:",
    X_base_multi.shape[1]
)



# ==================================================
# 6. Signature 함수
# ==================================================
def make_centroids(
    X_bin,
    labels,
    reference_idx,
    n_classes
):
    centroids = np.zeros(
        (n_classes, X_bin.shape[1]),
        dtype=np.float32
    )

    for c in range(n_classes):

        class_rows = reference_idx[
            labels[reference_idx] == c
        ]

        if len(class_rows) > 0:
            centroids[c] = np.asarray(
                X_bin[class_rows].mean(axis=0)
            ).ravel().astype(np.float32)

    return centroids


def cosine_signature(
    X_rows,
    centroids
):
    dot = X_rows @ centroids.T

    dot = np.asarray(
        dot,
        dtype=np.float32
    )

    patient_norm = np.sqrt(
        np.asarray(
            X_rows.multiply(X_rows)
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


# ==================================================
# 6-1. 26-class Global One-vs-Rest Log-Odds
# ==================================================
def make_global_logodds(
    X_bin,
    labels,
    reference_idx,
    n_classes,
    alpha=2.0,
):
    """
    reference_idx의 TRAIN 데이터만 사용한다.

    각 암종 c에 대해:
        c 암종
        vs
        나머지 25개 암종

    의 변이별 log-odds 차이를 계산한다.

    반환 shape:
        (n_classes, n_genes)
    """

    weights = np.zeros(
        (
            n_classes,
            X_bin.shape[1]
        ),
        dtype=np.float32
    )

    total_count = np.asarray(
        X_bin[reference_idx].sum(axis=0)
    ).ravel().astype(np.float32)

    n_total = len(reference_idx)

    for c in range(n_classes):

        class_rows = reference_idx[
            labels[reference_idx] == c
        ]

        class_count = np.asarray(
            X_bin[class_rows].sum(axis=0)
        ).ravel().astype(np.float32)

        rest_count = (
            total_count - class_count
        )

        n_class = max(
            len(class_rows),
            1
        )

        n_rest = max(
            n_total - len(class_rows),
            1
        )

        p_class = (
            class_count + alpha
        ) / (
            n_class + 2.0 * alpha
        )

        p_rest = (
            rest_count + alpha
        ) / (
            n_rest + 2.0 * alpha
        )

        logit_class = np.log(
            p_class / (1.0 - p_class)
        )

        logit_rest = np.log(
            p_rest / (1.0 - p_rest)
        )

        class_weights = (
            logit_class - logit_rest
        ).astype(np.float32)

        # reference TRAIN에서 한 번도 관찰되지 않은 변이는
        # 정보가 없으므로 0점
        unseen = total_count == 0

        class_weights[unseen] = 0.0

        weights[c] = class_weights

    return weights


def global_logodds_score(
    X_rows,
    weights,
):
    """
    환자별로 26개 암종에 대한
    supervised mutation enrichment 점수를 만든다.

    반환 shape:
        (n_patients, n_classes)
    """

    raw_score = np.asarray(
        X_rows @ weights.T
    ).astype(np.float32)

    mutation_n = np.asarray(
        X_rows.sum(axis=1)
    ).ravel().astype(np.float32)

    return np.divide(
        raw_score,
        np.maximum(
            mutation_n[:, None],
            1.0
        ),
    ).astype(np.float32)


# ==================================================
# 6-1. LGG ↔ GBMLGG Pair Log-Odds
# ==================================================
def make_pair_logodds(
    X_bin,
    labels,
    reference_idx,
    lgg_idx,
    gbmlgg_idx,
    alpha=2.0,
):
    """
    reference_idx 안의 TRAIN 데이터만 사용해서
    각 변이가 LGG 쪽인지 GBMLGG 쪽인지
    log-odds 가중치를 계산한다.
    """

    lgg_rows = reference_idx[
        labels[reference_idx] == lgg_idx
    ]

    gbmlgg_rows = reference_idx[
        labels[reference_idx] == gbmlgg_idx
    ]

    lgg_count = np.asarray(
        X_bin[lgg_rows].sum(axis=0)
    ).ravel().astype(np.float32)

    gbmlgg_count = np.asarray(
        X_bin[gbmlgg_rows].sum(axis=0)
    ).ravel().astype(np.float32)

    n_lgg = max(len(lgg_rows), 1)
    n_gbmlgg = max(len(gbmlgg_rows), 1)

    p_lgg = (
        lgg_count + alpha
    ) / (
        n_lgg + 2.0 * alpha
    )

    p_gbmlgg = (
        gbmlgg_count + alpha
    ) / (
        n_gbmlgg + 2.0 * alpha
    )

    logit_lgg = np.log(
        p_lgg / (1.0 - p_lgg)
    )

    logit_gbmlgg = np.log(
        p_gbmlgg / (1.0 - p_gbmlgg)
    )

    weights = (
        logit_lgg
        - logit_gbmlgg
    ).astype(np.float32)

    # 두 암종에서 한 번도 관찰되지 않은 변이는
    # 정보가 없으므로 가중치 0
    unseen = (
        lgg_count + gbmlgg_count
    ) == 0

    weights[unseen] = 0.0

    return weights


def pair_logodds_score(
    X_rows,
    weights,
):
    """
    환자 한 명의 여러 변이 log-odds를
    평균내어 최종 숫자 1개로 만든다.
    """

    raw_score = np.asarray(
        X_rows @ weights
    ).ravel().astype(np.float32)

    mutation_n = np.asarray(
        X_rows.sum(axis=1)
    ).ravel().astype(np.float32)

    return np.divide(
        raw_score,
        np.maximum(mutation_n, 1.0),
    ).astype(np.float32)



# ==================================================

# 7. 모델
# ==================================================
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


# ==================================================
# 8. TEST 데이터 준비
# ==================================================
test = pd.read_csv("test.csv").reset_index(drop=True)
sample = pd.read_csv("sample_submission.csv")

print("\nTEST 환자 수:", len(test))


# Binary mutation
binary_test_np = (
    test[genes].notna()
    & test[genes].ne("WT")
).to_numpy(dtype=np.uint8)

X_binary_test = sparse.csr_matrix(
    binary_test_np,
    dtype=np.float32
)

mutation_count_test = (
    binary_test_np.sum(axis=1)
    .astype(np.float32)
    .reshape(-1, 1)
)


# mutation type + gene_position + NLP document
type_counts_test = np.zeros(
    (len(test), 5),
    dtype=np.float32
)

position_tokens_test = [
    [] for _ in range(len(test))
]

document_tokens_test = [
    [] for _ in range(len(test))
]

mutations_test = test[genes].stack().dropna()
mutations_test = mutations_test[
    mutations_test.ne("WT")
].astype(str)

for (row_idx, gene), mutation in mutations_test.items():

    if "fs" in mutation.lower():
        type_counts_test[row_idx, 2] += 1

    elif "*" in mutation:
        type_counts_test[row_idx, 3] += 1

    else:
        aa = re.search(
            r"([A-Z])(\d+)([A-Z])",
            mutation
        )

        if aa:
            before, _, after = aa.groups()

            if before == after:
                type_counts_test[row_idx, 1] += 1
            else:
                type_counts_test[row_idx, 0] += 1

        else:
            type_counts_test[row_idx, 4] += 1

    pos = re.search(r"(\d+)", mutation)

    if pos:
        position = int(pos.group(1))

        position_tokens_test[row_idx].append(
            f"{gene}_POS{position}"
        )

    document_tokens_test[row_idx].append(
        f"{gene}_{mutation}"
    )


documents_test = [
    " ".join(tokens)
    for tokens in document_tokens_test
]

X_position_test = hasher.transform(
    position_tokens_test
)

X_base_test = sparse.hstack(
    [
        X_binary_test,
        sparse.csr_matrix(mutation_count_test),
        sparse.csr_matrix(type_counts_test),
        X_position_test,
    ],
    format="csr"
)

print(
    "TEST BEST 기본 피처 수:",
    X_base_test.shape[1]
)

# ==================================================
# TEST용 BEST07 Multi-Mutation X_base
# exp32는 기존 X_base_test 유지
# ==================================================
type_counts_test_multi = np.zeros(
    (len(test), 5),
    dtype=np.float32
)

position_tokens_test_multi = [
    [] for _ in range(len(test))
]

for (row_idx, gene), mutation in mutations_test.items():

    for mutation_token in str(mutation).split():

        if "fs" in mutation_token.lower():
            type_counts_test_multi[row_idx, 2] += 1

        elif "*" in mutation_token:
            type_counts_test_multi[row_idx, 3] += 1

        else:
            aa = re.search(
                r"([A-Z])(\d+)([A-Z])",
                mutation_token
            )

            if aa:
                before, _, after = aa.groups()

                if before == after:
                    type_counts_test_multi[row_idx, 1] += 1
                else:
                    type_counts_test_multi[row_idx, 0] += 1

            else:
                type_counts_test_multi[row_idx, 4] += 1

        pos = re.search(
            r"(\d+)",
            mutation_token
        )

        if pos:
            position = int(pos.group(1))

            position_tokens_test_multi[row_idx].append(
                f"{gene}_POS{position}"
            )

X_position_test_multi = hasher.transform(
    position_tokens_test_multi
)

X_base_test_multi = sparse.hstack(
    [
        X_binary_test,
        sparse.csr_matrix(mutation_count_test),
        sparse.csr_matrix(type_counts_test_multi),
        X_position_test_multi,
    ],
    format="csr"
)

print(
    "BEST07 TEST Multi-parser 기본 피처 수:",
    X_base_test_multi.shape[1]
)


print(
    "TEST NLP 문서 수:",
    len(documents_test)
)

print("\n✅ BEST_02 TEST 피처 준비 코드 완료")

# ==================================================
# ==================================================
# 9. BEST_06 Candidate TEST 예측
#
# 각 Seed마다:
#   A. BEST_05 예측
#   B. exp_32 예측
#   C. Frozen Router 적용
#
# 마지막:
#   3 Seed Router 다수결
#   3개가 모두 다르면 BEST_05 base 80:20
#   3-Seed 평균 확률 argmax를 fallback
# ==================================================

seeds = [42, 123, 2026]

all_train_idx = np.arange(len(train))

lgg_idx = le.transform(["LGG"])[0]
gbmlgg_idx = le.transform(["GBMLGG"])[0]
kipan_idx = le.transform(["KIPAN"])[0]
kirc_idx = le.transform(["KIRC"])[0]

router_seed_predictions = []
test_prediction_rows = []

# 3 Seed가 모두 다른 경우 사용할 fallback
best05_base_proba_sum = np.zeros(
    (len(test), n_classes),
    dtype=np.float64
)


for seed in seeds:

    print("\n" + "=" * 70)
    print("🌱 BEST_07 CHECK 제출 Seed:", seed)
    print("=" * 70)

    # ==================================================
    # A. TRAIN용 target-derived feature
    #    5-Fold cross-fitting
    # ==================================================
    signature_train = np.zeros(
        (len(train), n_classes),
        dtype=np.float32
    )

    global_logodds_train = np.zeros(
        (len(train), n_classes),
        dtype=np.float32
    )

    lgg_pair_logodds_train = np.zeros(
        len(train),
        dtype=np.float32
    )

    kidney_pair_logodds_train = np.zeros(
        len(train),
        dtype=np.float32
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    for reference_idx, valid_idx in cv.split(
        np.zeros(len(train)),
        y
    ):

        # Signature
        centroids_fold = make_centroids(
            X_binary,
            y,
            reference_idx,
            n_classes
        )

        signature_train[valid_idx] = cosine_signature(
            X_binary[valid_idx],
            centroids_fold
        )

        # 26-class Global Log-Odds
        global_weights_fold = make_global_logodds(
            X_binary,
            y,
            reference_idx,
            n_classes,
            alpha=2.0
        )

        global_logodds_train[valid_idx] = (
            global_logodds_score(
                X_binary[valid_idx],
                global_weights_fold
            )
        )

        # LGG ↔ GBMLGG Pair Log-Odds
        lgg_weights_fold = make_pair_logodds(
            X_binary,
            y,
            reference_idx,
            lgg_idx,
            gbmlgg_idx,
            alpha=2.0
        )

        lgg_pair_logodds_train[valid_idx] = (
            pair_logodds_score(
                X_binary[valid_idx],
                lgg_weights_fold
            )
        )

        # KIPAN ↔ KIRC Pair Log-Odds
        kidney_weights_fold = make_pair_logodds(
            X_binary,
            y,
            reference_idx,
            kipan_idx,
            kirc_idx,
            alpha=2.0
        )

        kidney_pair_logodds_train[valid_idx] = (
            pair_logodds_score(
                X_binary[valid_idx],
                kidney_weights_fold
            )
        )


    # ==================================================
    # B. TEST용 target-derived feature
    #    전체 TRAIN만 사용
    # ==================================================

    # Signature
    centroids_full = make_centroids(
        X_binary,
        y,
        all_train_idx,
        n_classes
    )

    signature_test = cosine_signature(
        X_binary_test,
        centroids_full
    )

    # Global Log-Odds
    global_weights_full = make_global_logodds(
        X_binary,
        y,
        all_train_idx,
        n_classes,
        alpha=2.0
    )

    global_logodds_test = global_logodds_score(
        X_binary_test,
        global_weights_full
    )

    # LGG ↔ GBMLGG
    lgg_weights_full = make_pair_logodds(
        X_binary,
        y,
        all_train_idx,
        lgg_idx,
        gbmlgg_idx,
        alpha=2.0
    )

    lgg_pair_logodds_test = pair_logodds_score(
        X_binary_test,
        lgg_weights_full
    )

    # KIPAN ↔ KIRC
    kidney_weights_full = make_pair_logodds(
        X_binary,
        y,
        all_train_idx,
        kipan_idx,
        kirc_idx,
        alpha=2.0
    )

    kidney_pair_logodds_test = pair_logodds_score(
        X_binary_test,
        kidney_weights_full
    )


    # ==================================================
    # C. NLP
    # BEST_05와 exp_32가 공통으로 사용
    # ==================================================
    vectorizer = make_vectorizer()

    X_nlp_train = vectorizer.fit_transform(
        documents
    )

    X_nlp_test = vectorizer.transform(
        documents_test
    )

    nlp_model = make_nlp(seed)

    nlp_calibrated = CalibratedClassifierCV(
        nlp_model,
        method="sigmoid",
        cv=3
    )

    nlp_calibrated.fit(
        X_nlp_train,
        y
    )

    nlp_proba = nlp_calibrated.predict_proba(
        X_nlp_test
    )


    # ==================================================
    # D. BEST_05 BASE
    #    X_base_multi + Signature
    # ==================================================
    X_best05_train = sparse.hstack(
        [
            X_base_multi,
            sparse.csr_matrix(signature_train)
        ],
        format="csr"
    )

    X_best05_test = sparse.hstack(
        [
            X_base_test_multi,
            sparse.csr_matrix(signature_test)
        ],
        format="csr"
    )

    best05_base_model = make_xgb(seed)

    best05_base_model.fit(
        X_best05_train,
        y
    )

    best05_xgb_proba = (
        best05_base_model.predict_proba(
            X_best05_test
        )
    )

    best05_ensemble_proba = (
        0.80 * best05_xgb_proba
        + 0.20 * nlp_proba
    )

    best05_base_proba_sum += (
        best05_ensemble_proba
    )

    best05_pred = np.argmax(
        best05_ensemble_proba,
        axis=1
    ).astype(np.int32)


    # ==================================================
    # E. BEST_05
    #    LGG ↔ GBMLGG Specialist
    # ==================================================
    X_best05_lgg_train = sparse.hstack(
        [
            X_base_multi,
            sparse.csr_matrix(signature_train),
            sparse.csr_matrix(
                lgg_pair_logodds_train.reshape(-1, 1)
            )
        ],
        format="csr"
    )

    X_best05_lgg_test = sparse.hstack(
        [
            X_base_test_multi,
            sparse.csr_matrix(signature_test),
            sparse.csr_matrix(
                lgg_pair_logodds_test.reshape(-1, 1)
            )
        ],
        format="csr"
    )

    lgg_y = np.full(
        len(train),
        2,
        dtype=np.int32
    )

    lgg_y[y == lgg_idx] = 0
    lgg_y[y == gbmlgg_idx] = 1

    lgg_counts = np.bincount(
        lgg_y,
        minlength=3
    )

    lgg_class_weights = (
        len(lgg_y)
        / (
            3.0
            * np.maximum(lgg_counts, 1)
        )
    )

    lgg_sample_weight = (
        lgg_class_weights[lgg_y]
    )

    best05_lgg_model = make_xgb(seed)

    best05_lgg_model.fit(
        X_best05_lgg_train,
        lgg_y,
        sample_weight=lgg_sample_weight
    )

    best05_final = best05_pred.copy()

    route_positions = np.where(
        np.isin(
            best05_final,
            [lgg_idx, gbmlgg_idx]
        )
    )[0]

    if len(route_positions) > 0:

        sp = best05_lgg_model.predict(
            X_best05_lgg_test[
                route_positions
            ]
        ).astype(np.int32)

        for pos, pred in zip(
            route_positions,
            sp
        ):

            if pred == 0:
                best05_final[pos] = lgg_idx

            elif pred == 1:
                best05_final[pos] = gbmlgg_idx


    # ==================================================
    # F. BEST_05
    #    KIPAN ↔ KIRC Specialist
    # ==================================================
    X_best05_kidney_train = sparse.hstack(
        [
            X_base_multi,
            sparse.csr_matrix(signature_train),
            sparse.csr_matrix(
                kidney_pair_logodds_train.reshape(-1, 1)
            )
        ],
        format="csr"
    )

    X_best05_kidney_test = sparse.hstack(
        [
            X_base_test_multi,
            sparse.csr_matrix(signature_test),
            sparse.csr_matrix(
                kidney_pair_logodds_test.reshape(-1, 1)
            )
        ],
        format="csr"
    )

    kidney_y = np.full(
        len(train),
        2,
        dtype=np.int32
    )

    kidney_y[y == kipan_idx] = 0
    kidney_y[y == kirc_idx] = 1

    kidney_counts = np.bincount(
        kidney_y,
        minlength=3
    )

    kidney_class_weights = (
        len(kidney_y)
        / (
            3.0
            * np.maximum(kidney_counts, 1)
        )
    )

    kidney_sample_weight = (
        kidney_class_weights[kidney_y]
    )

    best05_kidney_model = make_xgb(seed)

    best05_kidney_model.fit(
        X_best05_kidney_train,
        kidney_y,
        sample_weight=kidney_sample_weight
    )

    kidney_positions = np.where(
        best05_final == kipan_idx
    )[0]

    if len(kidney_positions) > 0:

        kp = best05_kidney_model.predict(
            X_best05_kidney_test[
                kidney_positions
            ]
        ).astype(np.int32)

        for pos, pred in zip(
            kidney_positions,
            kp
        ):

            if pred == 1:
                best05_final[pos] = kirc_idx


    # ==================================================
    # G. exp_32 Global Base
    #    X_base + Signature + Global Log-Odds
    # ==================================================
    X_exp32_train = sparse.hstack(
        [
            X_base,
            sparse.csr_matrix(signature_train),
            sparse.csr_matrix(global_logodds_train)
        ],
        format="csr"
    )

    X_exp32_test = sparse.hstack(
        [
            X_base_test,
            sparse.csr_matrix(signature_test),
            sparse.csr_matrix(global_logodds_test)
        ],
        format="csr"
    )

    exp32_base_model = make_xgb(seed)

    exp32_base_model.fit(
        X_exp32_train,
        y
    )

    exp32_xgb_proba = (
        exp32_base_model.predict_proba(
            X_exp32_test
        )
    )

    exp32_ensemble_proba = (
        0.80 * exp32_xgb_proba
        + 0.20 * nlp_proba
    )

    exp32_final = np.argmax(
        exp32_ensemble_proba,
        axis=1
    ).astype(np.int32)


    # ==================================================
    # H. exp_32
    #    LGG ↔ GBMLGG Specialist
    # ==================================================
    X_exp32_lgg_train = sparse.hstack(
        [
            X_base,
            sparse.csr_matrix(signature_train),
            sparse.csr_matrix(global_logodds_train),
            sparse.csr_matrix(
                lgg_pair_logodds_train.reshape(-1, 1)
            )
        ],
        format="csr"
    )

    X_exp32_lgg_test = sparse.hstack(
        [
            X_base_test,
            sparse.csr_matrix(signature_test),
            sparse.csr_matrix(global_logodds_test),
            sparse.csr_matrix(
                lgg_pair_logodds_test.reshape(-1, 1)
            )
        ],
        format="csr"
    )

    exp32_lgg_model = make_xgb(seed)

    exp32_lgg_model.fit(
        X_exp32_lgg_train,
        lgg_y,
        sample_weight=lgg_sample_weight
    )

    route_positions = np.where(
        np.isin(
            exp32_final,
            [lgg_idx, gbmlgg_idx]
        )
    )[0]

    if len(route_positions) > 0:

        sp = exp32_lgg_model.predict(
            X_exp32_lgg_test[
                route_positions
            ]
        ).astype(np.int32)

        for pos, pred in zip(
            route_positions,
            sp
        ):

            if pred == 0:
                exp32_final[pos] = lgg_idx

            elif pred == 1:
                exp32_final[pos] = gbmlgg_idx


    # ==================================================
    # I. exp_32
    #    KIPAN ↔ KIRC Specialist
    # ==================================================
    X_exp32_kidney_train = sparse.hstack(
        [
            X_base,
            sparse.csr_matrix(signature_train),
            sparse.csr_matrix(global_logodds_train),
            sparse.csr_matrix(
                kidney_pair_logodds_train.reshape(-1, 1)
            )
        ],
        format="csr"
    )

    X_exp32_kidney_test = sparse.hstack(
        [
            X_base_test,
            sparse.csr_matrix(signature_test),
            sparse.csr_matrix(global_logodds_test),
            sparse.csr_matrix(
                kidney_pair_logodds_test.reshape(-1, 1)
            )
        ],
        format="csr"
    )

    exp32_kidney_model = make_xgb(seed)

    exp32_kidney_model.fit(
        X_exp32_kidney_train,
        kidney_y,
        sample_weight=kidney_sample_weight
    )

    kidney_positions = np.where(
        exp32_final == kipan_idx
    )[0]

    if len(kidney_positions) > 0:

        kp = exp32_kidney_model.predict(
            X_exp32_kidney_test[
                kidney_positions
            ]
        ).astype(np.int32)

        for pos, pred in zip(
            kidney_positions,
            kp
        ):

            if pred == 1:
                exp32_final[pos] = kirc_idx


    # ==================================================
    # J. Frozen Router
    # ==================================================
    router_pred = best05_final.copy()

    rule1 = (
        (best05_final == kirc_idx)
        & (exp32_final == kipan_idx)
    )

    rule2 = (
        (best05_final == gbmlgg_idx)
        & (exp32_final == lgg_idx)
    )

    rule3 = (
        (best05_final == lgg_idx)
        & (exp32_final == gbmlgg_idx)
    )

    use_exp32 = (
        rule1 | rule2 | rule3
    )

    router_pred[use_exp32] = (
        exp32_final[use_exp32]
    )

    router_seed_predictions.append(
        router_pred.copy()
    )

    print(
        "BEST_05 ↔ exp_32 disagreement:",
        int((best05_final != exp32_final).sum())
    )

    print(
        "Frozen Router 변경:",
        int(use_exp32.sum())
    )

    # Seed별 TEST 예측 기록
    best05_names = le.inverse_transform(
        best05_final
    )

    exp32_names = le.inverse_transform(
        exp32_final
    )

    router_names = le.inverse_transform(
        router_pred
    )

    for i in range(len(test)):

        test_prediction_rows.append({
            "ID": test.loc[i, "ID"],
            "seed": seed,
            "best05_pred": best05_names[i],
            "exp32_pred": exp32_names[i],
            "router_pred": router_names[i],
        })


# ==================================================
# 10. 3 Seed Frozen Router 결합
# ==================================================
router_matrix = np.vstack(
    router_seed_predictions
)

fallback_proba = (
    best05_base_proba_sum
    / len(seeds)
)

fallback_pred = np.argmax(
    fallback_proba,
    axis=1
).astype(np.int32)

final_pred = np.empty(
    len(test),
    dtype=np.int32
)

majority_count = 0
fallback_count = 0

for i in range(len(test)):

    votes = router_matrix[:, i]

    counts = np.bincount(
        votes,
        minlength=n_classes
    )

    winner = int(
        np.argmax(counts)
    )

    if counts[winner] >= 2:

        final_pred[i] = winner
        majority_count += 1

    else:

        final_pred[i] = fallback_pred[i]
        fallback_count += 1


final_subclass = le.inverse_transform(
    final_pred
)


# ==================================================
# 11. TEST 예측 기록 저장
# ==================================================
test_prediction_df = pd.DataFrame(
    test_prediction_rows
)

test_prediction_df.to_csv(
    "BEST_07_multi_mutation_router_test_seed_predictions.csv",
    index=False
)


# ==================================================
# 12. Dacon 제출 CSV
# ==================================================
submission = sample.copy()

submission["ID"] = test["ID"]
submission["SUBCLASS"] = final_subclass

assert len(submission) == len(test)
assert len(submission) == 2546

assert submission.columns.tolist() == [
    "ID",
    "SUBCLASS"
]

assert submission["SUBCLASS"].notna().all()

submission.to_csv(
    "aeyoung_BEST07_CHECK_multi_mutation_router.csv",
    index=False
)


print("\n" + "=" * 70)
print("✅ BEST_07 CHECK Multi-Mutation Router 제출 파일 생성 완료")
print("=" * 70)

print(
    "파일:",
    "aeyoung_BEST07_CHECK_multi_mutation_router.csv"
)

print(
    "Seed별 예측 기록:",
    "BEST_07_multi_mutation_router_test_seed_predictions.csv"
)

print("행 수:", len(submission))

print(
    "3-Seed 다수결 적용:",
    majority_count
)

print(
    "3개 Seed 모두 달라 fallback:",
    fallback_count
)

print("\n암종 예측 분포:")

print(
    submission["SUBCLASS"]
    .value_counts()
    .to_string()
)
