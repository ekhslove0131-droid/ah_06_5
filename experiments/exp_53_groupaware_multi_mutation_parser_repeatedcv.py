from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
from sklearn.metrics import f1_score


# ============================================================
# exp_53 | Group-aware Multi-Mutation Parser Repeated CV
#
# BASELINE:
#   기존 exp_38a 결과 그대로 사용
#
# CANDIDATE:
#   exp_38a 코드에서 오직 mutation type / position parser만
#   whitespace-separated mutation token 단위로 변경
#
# 공통:
#   - Exact mutation groups 동일
#   - Outer StratifiedGroupKFold 동일
#   - Inner signature StratifiedGroupKFold 동일
#   - NLP calibration StratifiedGroupKFold 동일
#   - Binary 동일
#   - mutation_count 동일 (mutated gene/cell count)
#   - Signature 동일
#   - NLP document 동일
#   - XGB 설정 동일
#
# TEST는 절대 읽지 않는다.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_SCRIPT = (
    BASE_DIR
    / "exp_38a_groupaware_base_newseeds.py"
)

BASE_SEED_FILE = (
    BASE_DIR
    / "exp_38a_groupaware_base_newseeds_seed_results.csv"
)

BASE_FOLD_FILE = (
    BASE_DIR
    / "exp_38a_groupaware_base_newseeds_fold_results.csv"
)

BASE_PRED_FILE = (
    BASE_DIR
    / "exp_38a_groupaware_base_newseeds_predictions.csv"
)

CANDIDATE_PREFIX = (
    "exp_53_candidate_groupaware_multi_mutation"
)

RUNNER_PATH = (
    BASE_DIR
    / "_exp_53_candidate_groupaware_runner.py"
)


# ============================================================
# 1. Safety checks
# ============================================================

required = [
    SOURCE_SCRIPT,
    BASE_SEED_FILE,
    BASE_FOLD_FILE,
    BASE_PRED_FILE,
]

for path in required:
    if not path.exists():
        raise FileNotFoundError(path)


source = SOURCE_SCRIPT.read_text(
    encoding="utf-8"
)

if "test.csv" in source.lower():
    raise RuntimeError(
        "Safety stop: source script에서 test.csv 문자열이 발견되었습니다."
    )


# ============================================================
# 2. Parser-only patch
#
# 기존:
#   한 gene cell의 전체 mutation 문자열을 한 번만 해석
#
# Candidate:
#   whitespace로 나눈 각 mutation token마다
#   기존과 동일한 type / first-position 규칙 적용
#
# NLP document는 아래 원래 코드가 그대로 실행되므로
# full mutation string을 유지한다.
# ============================================================

pattern = re.compile(
    r"(?ms)"
    r"^[ \t]*# --------------------------\n"
    r"[ \t]*# mutation type\n"
    r"[ \t]*# --------------------------\n"
    r".*?"
    r"^[ \t]*# --------------------------\n"
    r"[ \t]*# NLP mutation document\n"
    r"[ \t]*# --------------------------\n"
)

replacement = '''    # --------------------------
    # mutation type + gene_position
    # Candidate: whitespace-separated event parsing
    # --------------------------
    for mutation_token in str(mutation).split():

        # ----------------------
        # mutation type
        # 기존 exp_38a 규칙을
        # 각 token에 독립 적용
        # ----------------------
        if "fs" in mutation_token.lower():
            type_counts[row_idx, 2] += 1

        elif "*" in mutation_token:
            type_counts[row_idx, 3] += 1

        else:
            aa = re.search(
                r"([A-Z])(\\d+)([A-Z])",
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

        # ----------------------
        # gene_position
        # 각 mutation token에서
        # 첫 숫자 위치를 독립 추출
        # ----------------------
        pos = re.search(
            r"(\\d+)",
            mutation_token
        )

        if pos:
            position = int(
                pos.group(1)
            )

            position_tokens[
                row_idx
            ].append(
                f"{gene}_POS{position}"
            )

            position_found += 1

    # --------------------------
    # NLP mutation document
    # 기존 full mutation string 그대로 유지
    # --------------------------
'''

patched_source, replacement_count = (
    pattern.subn(
        lambda _match: replacement,
        source,
        count=1,
    )
)

if replacement_count != 1:
    raise RuntimeError(
        "Parser block을 정확히 1회 교체하지 못했습니다. "
        f"교체 횟수={replacement_count}"
    )


# ============================================================
# 3. Candidate 전용 파일명으로 변경
#    기존 exp_38a 결과 절대 덮어쓰지 않음
# ============================================================

patched_source = patched_source.replace(
    "exp_38a_groupaware_base_newseeds",
    CANDIDATE_PREFIX,
)

patched_source = patched_source.replace(
    "🧬 exp_38a | Group-aware BEST 80% + NLP 20%",
    "🧬 exp_53 candidate | Group-aware Multi-Mutation Parser",
)

patched_source = patched_source.replace(
    "📊 exp_38a | Group-aware Base 80:20 Validation",
    "📊 exp_53 candidate | Group-aware Multi-Mutation Validation",
)


# ============================================================
# 4. Candidate runner 생성 및 실행
# ============================================================

RUNNER_PATH.write_text(
    patched_source,
    encoding="utf-8",
)

print("=" * 80)
print("exp_53 | GROUP-AWARE MULTI-MUTATION PARSER")
print("=" * 80)
print("Baseline : existing exp_38a")
print("Candidate: parser-only change")
print("Seeds    : 7 / 77 / 777")
print("Outer CV : StratifiedGroupKFold")
print("Inner CV : StratifiedGroupKFold")
print("NLP CV   : StratifiedGroupKFold")
print("TEST FILE: NEVER READ")
print("=" * 80)

try:
    subprocess.run(
        [
            sys.executable,
            "-u",
            str(RUNNER_PATH),
        ],
        cwd=BASE_DIR,
        check=True,
    )

finally:
    if RUNNER_PATH.exists():
        RUNNER_PATH.unlink()


# ============================================================
# 5. Load baseline / candidate predictions
# ============================================================

baseline_pred = pd.read_csv(
    BASE_PRED_FILE
)

candidate_pred = pd.read_csv(
    BASE_DIR
    / f"{CANDIDATE_PREFIX}_predictions.csv"
)

keys = [
    "seed",
    "row_index",
]

merged = baseline_pred.merge(
    candidate_pred,
    on=keys,
    how="inner",
    suffixes=(
        "_baseline",
        "_candidate",
    ),
    validate="one_to_one",
)


# ============================================================
# 6. Fair-comparison audits
# ============================================================

expected_rows = (
    len(baseline_pred)
)

row_count_exact = (
    len(merged)
    == expected_rows
    == len(candidate_pred)
)

fold_assignment_identical = bool(
    (
        merged[
            "fold_baseline"
        ]
        == merged[
            "fold_candidate"
        ]
    ).all()
)

id_identical = bool(
    (
        merged[
            "ID_baseline"
        ]
        == merged[
            "ID_candidate"
        ]
    ).all()
)

truth_identical = bool(
    (
        merged[
            "true_subclass_baseline"
        ]
        == merged[
            "true_subclass_candidate"
        ]
    ).all()
)

nlp_hard_identical = bool(
    (
        merged[
            "nlp_pred_baseline"
        ]
        == merged[
            "nlp_pred_candidate"
        ]
    ).all()
)

fair_comparison = bool(
    row_count_exact
    and fold_assignment_identical
    and id_identical
    and truth_identical
    and nlp_hard_identical
)


# ============================================================
# 7. Seed-level comparison
# ============================================================

seed_rows = []
fold_rows = []

for seed in [
    7,
    77,
    777,
]:

    seed_data = merged[
        merged["seed"] == seed
    ].copy()

    truth = seed_data[
        "true_subclass_baseline"
    ]

    baseline_xgb_f1 = f1_score(
        truth,
        seed_data[
            "best_pred_baseline"
        ],
        average="macro",
    )

    candidate_xgb_f1 = f1_score(
        truth,
        seed_data[
            "best_pred_candidate"
        ],
        average="macro",
    )

    baseline_ensemble_f1 = f1_score(
        truth,
        seed_data[
            "ensemble_pred_baseline"
        ],
        average="macro",
    )

    candidate_ensemble_f1 = f1_score(
        truth,
        seed_data[
            "ensemble_pred_candidate"
        ],
        average="macro",
    )

    baseline_correct = (
        seed_data[
            "ensemble_pred_baseline"
        ]
        == truth
    )

    candidate_correct = (
        seed_data[
            "ensemble_pred_candidate"
        ]
        == truth
    )

    rescue = int(
        (
            ~baseline_correct
            & candidate_correct
        ).sum()
    )

    damage = int(
        (
            baseline_correct
            & ~candidate_correct
        ).sum()
    )

    seed_rows.append(
        {
            "seed": seed,
            "baseline_xgb_oof_f1":
                baseline_xgb_f1,
            "candidate_xgb_oof_f1":
                candidate_xgb_f1,
            "xgb_diff":
                candidate_xgb_f1
                - baseline_xgb_f1,
            "baseline_80_20_oof_f1":
                baseline_ensemble_f1,
            "candidate_80_20_oof_f1":
                candidate_ensemble_f1,
            "ensemble_diff":
                candidate_ensemble_f1
                - baseline_ensemble_f1,
            "candidate_rescue":
                rescue,
            "candidate_damage":
                damage,
            "net_correct_gain":
                rescue - damage,
        }
    )

    for fold in range(
        1,
        6,
    ):

        fold_data = seed_data[
            seed_data[
                "fold_baseline"
            ] == fold
        ]

        fold_truth = fold_data[
            "true_subclass_baseline"
        ]

        baseline_fold_xgb = f1_score(
            fold_truth,
            fold_data[
                "best_pred_baseline"
            ],
            average="macro",
        )

        candidate_fold_xgb = f1_score(
            fold_truth,
            fold_data[
                "best_pred_candidate"
            ],
            average="macro",
        )

        baseline_fold_ensemble = f1_score(
            fold_truth,
            fold_data[
                "ensemble_pred_baseline"
            ],
            average="macro",
        )

        candidate_fold_ensemble = f1_score(
            fold_truth,
            fold_data[
                "ensemble_pred_candidate"
            ],
            average="macro",
        )

        fold_rows.append(
            {
                "seed": seed,
                "fold": fold,
                "baseline_xgb_f1":
                    baseline_fold_xgb,
                "candidate_xgb_f1":
                    candidate_fold_xgb,
                "xgb_diff":
                    candidate_fold_xgb
                    - baseline_fold_xgb,
                "baseline_80_20_f1":
                    baseline_fold_ensemble,
                "candidate_80_20_f1":
                    candidate_fold_ensemble,
                "ensemble_diff":
                    candidate_fold_ensemble
                    - baseline_fold_ensemble,
            }
        )


seed_df = pd.DataFrame(
    seed_rows
)

fold_df = pd.DataFrame(
    fold_rows
)


# ============================================================
# 8. Save comparison
# ============================================================

seed_df.to_csv(
    BASE_DIR
    / "exp_53_groupaware_multi_mutation_seed_results.csv",
    index=False,
)

fold_df.to_csv(
    BASE_DIR
    / "exp_53_groupaware_multi_mutation_fold_results.csv",
    index=False,
)

merged.to_csv(
    BASE_DIR
    / "exp_53_groupaware_multi_mutation_predictions.csv",
    index=False,
)


# ============================================================
# 9. Final summary
# ============================================================

positive_seeds = int(
    (
        seed_df[
            "ensemble_diff"
        ] > 0
    ).sum()
)

positive_folds = int(
    (
        fold_df[
            "ensemble_diff"
        ] > 0
    ).sum()
)

mean_baseline = float(
    seed_df[
        "baseline_80_20_oof_f1"
    ].mean()
)

mean_candidate = float(
    seed_df[
        "candidate_80_20_oof_f1"
    ].mean()
)

mean_diff = float(
    seed_df[
        "ensemble_diff"
    ].mean()
)

mean_xgb_diff = float(
    seed_df[
        "xgb_diff"
    ].mean()
)


print(
    "\n"
    + "=" * 80
)

print(
    "exp_53 GROUP-AWARE PARSER COMPARISON"
)

print(
    "=" * 80
)

print(
    "동일 row 수:",
    row_count_exact,
)

print(
    "동일 fold 배정:",
    fold_assignment_identical,
)

print(
    "동일 ID:",
    id_identical,
)

print(
    "동일 정답:",
    truth_identical,
)

print(
    "동일 NLP hard prediction:",
    nlp_hard_identical,
)

print(
    "Fair comparison audit:",
    fair_comparison,
)

print()

print(
    seed_df.to_string(
        index=False
    )
)

print()

print(
    "개선 Seed:",
    f"{positive_seeds}/3",
)

print(
    "개선 Fold:",
    f"{positive_folds}/15",
)

print(
    "평균 Baseline 80:20:",
    round(
        mean_baseline,
        6,
    ),
)

print(
    "평균 Candidate 80:20:",
    round(
        mean_candidate,
        6,
    ),
)

print(
    "평균 앙상블 차이:",
    f"{mean_diff:+.6f}",
)

print(
    "평균 XGB 차이:",
    f"{mean_xgb_diff:+.6f}",
)

print()

if not fair_comparison:
    print(
        "RESULT: HOLD ⚠️ "
        "공정 비교 audit 실패 — parser 효과로 해석 금지"
    )
else:
    print(
        "RESULT: FAIR COMPARISON COMPLETE ✅"
    )

print(
    "Group-aware 결과는 효과의 강건성 검증용이며 "
    "Public LB 추정치가 아닙니다."
)

print(
    "TEST FILE: NEVER READ"
)

print(
    "=" * 80
)
