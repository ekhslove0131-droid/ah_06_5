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
    # mutation type + gene_position
    # Multi-mutation parser:
    # 한 gene cell 안의 whitespace-separated mutation을
    # 각각 독립적인 event로 해석
    # --------------------------
    for mutation_token in str(mutation).split():

        # mutation type
        if "fs" in mutation_token.lower():
            type_counts[row_idx, 2] += 1

        elif "*" in mutation_token:
            type_counts[row_idx, 3] += 1

        else:
            aa = re.search(
                r"([A-Z])(\d+)([A-Z])",
                mutation_token
            )

            if aa:
                before, _, after = aa.groups()

                if before == after:
                    type_counts[row_idx, 1] += 1
                else:
                    type_counts[row_idx, 0] += 1

            else:
                type_counts[row_idx, 4] += 1

        # gene_position
        pos = re.search(
            r"(\d+)",
            mutation_token
        )

        if pos:
            position = int(
                pos.group(1)
            )

            position_tokens[row_idx].append(
                f"{gene}_POS{position}"
            )

            position_found += 1

    # --------------------------
    # NLP mutation document
    # 기존 full mutation string 그대로 유지
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
# 8. LGG ↔ GBMLGG Specialist Pilot
#
# 기존 exp_06 OOF 예측은 그대로 사용하고
# Specialist만 같은 Fold에서 새로 학습한다.
#
# Specialist:
# 0 = LGG
# 1 = GBMLGG
# 2 = OTHER (기존 예측 유지)
# ==================================================

base_predictions = pd.read_csv(
    "exp_52_multi_mutation_predictions.csv"
)

lgg_idx = int(
    le.transform(["LGG"])[0]
)

gbmlgg_idx = int(
    le.transform(["GBMLGG"])[0]
)

kipan_idx = int(
    le.transform(["KIPAN"])[0]
)

kirc_idx = int(
    le.transform(["KIRC"])[0]
)

seeds = [7, 77, 777]

seed_rows = []
fold_rows = []
prediction_rows = []


for seed in seeds:

    print("\n" + "=" * 70)
    print("🌱 Seed:", seed)
    print("=" * 70)

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed
    )

    baseline_oof = np.zeros(
        len(train),
        dtype=np.int32
    )

    specialist_oof = np.zeros(
        len(train),
        dtype=np.int32
    )


    for fold, (train_idx, valid_idx) in enumerate(
        outer_cv.split(
            np.zeros(len(train)),
            y
        ),
        start=1
    ):

        print(
            f"\nSeed {seed} | Fold {fold}/5"
        )


        # ------------------------------------------
        # A. 기존 exp_06 앙상블 OOF 불러오기
        # ------------------------------------------
        base_fold = base_predictions[
            (base_predictions["seed"] == seed)
            & (base_predictions["fold"] == fold)
        ].set_index("row_index")

        assert set(
            base_fold.index
        ) == set(valid_idx)

        baseline_labels = (
            base_fold
            .loc[valid_idx, "candidate_ensemble_pred"]
            .to_numpy()
        )

        baseline_pred = le.transform(
            baseline_labels
        ).astype(np.int32)


        # ------------------------------------------
        # B. Specialist용 Signature
        # outer TRAIN은 cross-fitting
        # VALID는 outer TRAIN centroid만 사용
        # ------------------------------------------
        signature_train = np.zeros(
            (
                len(train_idx),
                n_classes
            ),
            dtype=np.float32
        )

        # LGG ↔ GBMLGG 전용 log-odds 점수
        # TRAIN 각 행은 자기 정답을 직접 보지 않도록
        # inner cross-fitting으로 생성
        pair_logodds_train = np.zeros(
            len(train_idx),
            dtype=np.float32
        )

        # KIPAN ↔ KIRC 전용 log-odds
        kidney_logodds_train = np.zeros(
            len(train_idx),
            dtype=np.float32
        )

        y_outer = y[train_idx]

        inner_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=seed
        )

        for inner_train_pos, inner_valid_pos in (
            inner_cv.split(
                np.zeros(len(train_idx)),
                y_outer
            )
        ):

            reference_idx = train_idx[
                inner_train_pos
            ]

            inner_valid_global = train_idx[
                inner_valid_pos
            ]

            centroids_inner = make_centroids(
                X_binary,
                y,
                reference_idx,
                n_classes
            )

            signature_train[
                inner_valid_pos
            ] = cosine_signature(
                X_binary[
                    inner_valid_global
                ],
                centroids_inner
            )

            # Pair Log-Odds도 같은 inner TRAIN만 사용
            pair_weights_inner = make_pair_logodds(
                X_binary,
                y,
                reference_idx,
                lgg_idx,
                gbmlgg_idx,
                alpha=2.0,
            )

            pair_logodds_train[
                inner_valid_pos
            ] = pair_logodds_score(
                X_binary[
                    inner_valid_global
                ],
                pair_weights_inner,
            )

            kidney_weights_inner = make_pair_logodds(
                X_binary,
                y,
                reference_idx,
                kipan_idx,
                kirc_idx,
                alpha=2.0,
            )

            kidney_logodds_train[
                inner_valid_pos
            ] = pair_logodds_score(
                X_binary[
                    inner_valid_global
                ],
                kidney_weights_inner,
            )


        centroids_outer = make_centroids(
            X_binary,
            y,
            train_idx,
            n_classes
        )

        signature_valid = cosine_signature(
            X_binary[valid_idx],
            centroids_outer
        )

        # outer VALID의 log-odds는
        # outer TRAIN 데이터만 보고 계산
        pair_weights_outer = make_pair_logodds(
            X_binary,
            y,
            train_idx,
            lgg_idx,
            gbmlgg_idx,
            alpha=2.0,
        )

        pair_logodds_valid = pair_logodds_score(
            X_binary[valid_idx],
            pair_weights_outer,
        )

        kidney_weights_outer = make_pair_logodds(
            X_binary,
            y,
            train_idx,
            kipan_idx,
            kirc_idx,
            alpha=2.0,
        )

        kidney_logodds_valid = pair_logodds_score(
            X_binary[valid_idx],
            kidney_weights_outer,
        )

        X_special_train = sparse.hstack(
            [
                X_base[train_idx],
                sparse.csr_matrix(
                    signature_train
                ),
                sparse.csr_matrix(
                    pair_logodds_train.reshape(-1, 1)
                ),
            ],
            format="csr"
        )

        X_special_valid = sparse.hstack(
            [
                X_base[valid_idx],
                sparse.csr_matrix(
                    signature_valid
                ),
                sparse.csr_matrix(
                    pair_logodds_valid.reshape(-1, 1)
                ),
            ],
            format="csr"
        )


        # ------------------------------------------
        # C. 3-class Specialist target
        # LGG / GBMLGG / OTHER
        # ------------------------------------------
        specialist_y = np.full(
            len(train_idx),
            2,
            dtype=np.int32
        )

        specialist_y[
            y[train_idx] == lgg_idx
        ] = 0

        specialist_y[
            y[train_idx] == gbmlgg_idx
        ] = 1


        # 클래스 불균형 보정
        counts = np.bincount(
            specialist_y,
            minlength=3
        )

        class_weights = (
            len(specialist_y)
            / (
                3.0
                * np.maximum(counts, 1)
            )
        )

        sample_weight = class_weights[
            specialist_y
        ]


        specialist_model = make_xgb(
            seed
        )

        specialist_model.fit(
            X_special_train,
            specialist_y,
            sample_weight=sample_weight
        )


        # ------------------------------------------
        # D. 기존 모델이 LGG/GBMLGG라고 한 환자만
        # Specialist에게 보낸다.
        # ------------------------------------------
        route_mask = np.isin(
            baseline_pred,
            [
                lgg_idx,
                gbmlgg_idx
            ]
        )

        final_pred = baseline_pred.copy()

        route_positions = np.where(
            route_mask
        )[0]

        if len(route_positions) > 0:

            specialist_pred = (
                specialist_model.predict(
                    X_special_valid[
                        route_positions
                    ]
                )
                .astype(np.int32)
            )

            for pos, sp in zip(
                route_positions,
                specialist_pred
            ):

                if sp == 0:
                    final_pred[pos] = lgg_idx

                elif sp == 1:
                    final_pred[pos] = gbmlgg_idx

                # sp == 2 (OTHER)
                # → 기존 예측 그대로 유지


        # ==========================================
        # D-2. KIPAN ↔ KIRC 3-class Specialist
        # 0 = KIPAN
        # 1 = KIRC
        # 2 = OTHER
        #
        # BEST_04가 KIPAN이라고 예측한 경우만
        # 다시 검사한다.
        # ==========================================
        best04_pred = final_pred.copy()

        X_kidney_train = sparse.hstack(
            [
                X_base[train_idx],
                sparse.csr_matrix(
                    signature_train
                ),
                sparse.csr_matrix(
                    kidney_logodds_train.reshape(-1, 1)
                ),
            ],
            format="csr"
        )

        X_kidney_valid = sparse.hstack(
            [
                X_base[valid_idx],
                sparse.csr_matrix(
                    signature_valid
                ),
                sparse.csr_matrix(
                    kidney_logodds_valid.reshape(-1, 1)
                ),
            ],
            format="csr"
        )

        kidney_y = np.full(
            len(train_idx),
            2,
            dtype=np.int32
        )

        kidney_y[
            y[train_idx] == kipan_idx
        ] = 0

        kidney_y[
            y[train_idx] == kirc_idx
        ] = 1

        kidney_counts = np.bincount(
            kidney_y,
            minlength=3
        )

        kidney_class_weights = (
            len(kidney_y)
            / (
                3.0
                * np.maximum(
                    kidney_counts,
                    1
                )
            )
        )

        kidney_sample_weight = (
            kidney_class_weights[
                kidney_y
            ]
        )

        kidney_model = make_xgb(
            seed
        )

        kidney_model.fit(
            X_kidney_train,
            kidney_y,
            sample_weight=kidney_sample_weight
        )

        kidney_route_mask = (
            best04_pred == kipan_idx
        )

        kidney_route_positions = np.where(
            kidney_route_mask
        )[0]

        if len(
            kidney_route_positions
        ) > 0:

            kidney_pred = (
                kidney_model.predict(
                    X_kidney_valid[
                        kidney_route_positions
                    ]
                )
                .astype(np.int32)
            )

            for pos, kp in zip(
                kidney_route_positions,
                kidney_pred
            ):

                # KIRC라고 판단한 경우에만 변경
                if kp == 1:
                    final_pred[pos] = kirc_idx

                # KIPAN 또는 OTHER이면
                # 기존 BEST_04 예측 유지


        # ------------------------------------------
        # E. Fold 평가
        # ------------------------------------------
        truth = y[valid_idx]

        baseline_f1 = f1_score(
            truth,
            best04_pred,
            average="macro"
        )

        specialist_f1 = f1_score(
            truth,
            final_pred,
            average="macro"
        )

        diff = (
            specialist_f1
            - baseline_f1
        )

        changed = int(
            np.sum(
                final_pred
                != best04_pred
            )
        )

        baseline_oof[
            valid_idx
        ] = best04_pred

        specialist_oof[
            valid_idx
        ] = final_pred

        fold_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "baseline_f1": baseline_f1,
                "specialist_f1": specialist_f1,
                "diff": diff,
                "routed": int(
                    route_mask.sum()
                ),
                "changed": changed,
            }
        )

        for pos, row_idx in enumerate(
            valid_idx
        ):

            prediction_rows.append(
                {
                    "seed": seed,
                    "fold": fold,
                    "row_index": int(row_idx),
                    "ID": train.loc[
                        row_idx,
                        "ID"
                    ],
                    "true_subclass":
                        le.inverse_transform(
                            [truth[pos]]
                        )[0],
                    "baseline_pred":
                        le.inverse_transform(
                            [best04_pred[pos]]
                        )[0],
                    "specialist_pred":
                        le.inverse_transform(
                            [final_pred[pos]]
                        )[0],
                }
            )

        print(
            f"BEST_04 F1 : "
            f"{baseline_f1:.5f}"
        )

        print(
            f"Specialist F1 : "
            f"{specialist_f1:.5f}"
        )

        print(
            f"차이          : "
            f"{diff:+.5f}"
        )

        print(
            f"라우팅 / 변경 : "
            f"{route_mask.sum()} / {changed}"
        )


    # ==============================================
    # Seed 전체 OOF
    # ==============================================
    baseline_seed_f1 = f1_score(
        y,
        baseline_oof,
        average="macro"
    )

    specialist_seed_f1 = f1_score(
        y,
        specialist_oof,
        average="macro"
    )

    seed_diff = (
        specialist_seed_f1
        - baseline_seed_f1
    )

    seed_rows.append(
        {
            "seed": seed,
            "baseline_oof_f1":
                baseline_seed_f1,
            "specialist_oof_f1":
                specialist_seed_f1,
            "diff":
                seed_diff,
        }
    )

    print("\n------------------------------------------")

    print(
        f"Seed {seed} 기존 OOF      : "
        f"{baseline_seed_f1:.5f}"
    )

    print(
        f"Seed {seed} Specialist OOF : "
        f"{specialist_seed_f1:.5f}"
    )

    print(
        f"Seed {seed} 차이           : "
        f"{seed_diff:+.5f}"
    )


# ==================================================
# 9. 최종 결과
# ==================================================
seed_df = pd.DataFrame(
    seed_rows
)

fold_df = pd.DataFrame(
    fold_rows
)

prediction_df = pd.DataFrame(
    prediction_rows
)

baseline_mean = seed_df[
    "baseline_oof_f1"
].mean()

specialist_mean = seed_df[
    "specialist_oof_f1"
].mean()

mean_diff = (
    specialist_mean
    - baseline_mean
)

improved_seeds = int(
    (seed_df["diff"] > 0).sum()
)

improved_folds = int(
    (fold_df["diff"] > 0).sum()
)


print("\n" + "=" * 75)
print("📊 exp_54 | Multi-Parser Base + Two Specialists")
print("=" * 75)

print(
    seed_df.to_string(
        index=False
    )
)

print(
    "\n기존 80:20 평균 F1:",
    round(
        baseline_mean,
        5
    )
)

print(
    "Specialist 평균 F1:",
    round(
        specialist_mean,
        5
    )
)

print(
    "평균 차이:",
    f"{mean_diff:+.5f}"
)

print(
    "개선 Seed:",
    improved_seeds,
    "/ 3"
)

print(
    "개선 Fold:",
    improved_folds,
    "/ 15"
)


# ==================================================
# 10. 저장
# ==================================================
seed_df.to_csv(
    "exp_34b_best05_newseeds_seed_results.csv",
    index=False
)

fold_df.to_csv(
    "exp_34b_best05_newseeds_fold_results.csv",
    index=False
)

prediction_df.to_csv(
    "exp_34b_best05_newseeds_predictions.csv",
    index=False
)

print("\n결과 저장 완료 ✅")

print(
    "exp_34b_best05_newseeds_seed_results.csv"
)

print(
    "exp_34b_best05_newseeds_fold_results.csv"
)

print(
    "exp_34b_best05_newseeds_predictions.csv"
)

print(
    "\n총 걸린 시간:",
    round(
        time.time()
        - start_time,
        2
    ),
    "초"
)
