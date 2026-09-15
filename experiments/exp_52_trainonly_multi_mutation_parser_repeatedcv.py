from pathlib import Path
import subprocess
import sys

import pandas as pd


# ============================================================
# exp_52 | Multi-Mutation Parser Repeated CV
#
# seed 7:
#   검증 완료된 exp_51 결과를 그대로 재사용
#
# seed 77 / 777:
#   exp_51과 동일한 코드를 seed만 변경하여 새로 실행
#
# 목적:
#   exp_51에서 발견된 multi-mutation parser 효과가
#   다른 Stratified CV seed에서도 반복되는지 확인
#
# TEST는 절대 읽지 않는다.
# BEST 파일은 수정하지 않는다.
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

SOURCE_SCRIPT = (
    BASE_DIR
    / "exp_51_trainonly_multi_mutation_parser_screen.py"
)

NEW_SEEDS = (77, 777)


def run_new_seed(seed: int) -> None:
    """
    검증된 exp_51 source를 그대로 가져와
    SEED와 출력 prefix만 변경한 임시 runner를 실행한다.
    """

    source = SOURCE_SCRIPT.read_text(
        encoding="utf-8"
    )

    seed_marker = "SEED = 7"

    if seed_marker not in source:
        raise RuntimeError(
            "exp_51에서 'SEED = 7'을 찾지 못했습니다."
        )

    source = source.replace(
        seed_marker,
        f"SEED = {seed}",
        1,
    )

    source = source.replace(
        "exp_51_multi_mutation_",
        f"exp_52_seed{seed}_multi_mutation_",
    )

    source = source.replace(
        "exp_51 | TRAIN-only Multi-Mutation Parser Screen",
        f"exp_52 seed {seed} | Multi-Mutation Parser Repeated CV",
    )

    runner_path = (
        BASE_DIR
        / f"_exp_52_seed{seed}_runner.py"
    )

    runner_path.write_text(
        source,
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 80,
        flush=True,
    )

    print(
        f"NEW VALIDATION SEED {seed}",
        flush=True,
    )

    print(
        "=" * 80,
        flush=True,
    )

    try:
        subprocess.run(
            [
                sys.executable,
                "-u",
                str(runner_path),
            ],
            cwd=BASE_DIR,
            check=True,
        )

    finally:
        if runner_path.exists():
            runner_path.unlink()


# ============================================================
# 1. Safety checks
# ============================================================

required_seed7_files = [
    "exp_51_multi_mutation_seed_results.csv",
    "exp_51_multi_mutation_fold_results.csv",
    "exp_51_multi_mutation_predictions.csv",
    "exp_51_multi_mutation_classwise_results.csv",
    "exp_51_multi_mutation_oof_probabilities.csv",
]

if not SOURCE_SCRIPT.exists():
    raise FileNotFoundError(
        SOURCE_SCRIPT
    )

for name in required_seed7_files:
    path = BASE_DIR / name

    if not path.exists():
        raise FileNotFoundError(
            path
        )


print("=" * 80)
print("exp_52 | Multi-Mutation Parser Repeated CV")
print("=" * 80)
print("Seed 7   : reuse exp_51")
print("Seed 77  : new training")
print("Seed 777 : new training")
print("TEST FILE: NEVER READ")
print("=" * 80)


# ============================================================
# 2. Train only new seeds
# ============================================================

for seed in NEW_SEEDS:
    run_new_seed(seed)


# ============================================================
# 3. Aggregate seed results
# ============================================================

seed_frames = [
    pd.read_csv(
        BASE_DIR
        / "exp_51_multi_mutation_seed_results.csv"
    )
]

for seed in NEW_SEEDS:
    seed_frames.append(
        pd.read_csv(
            BASE_DIR
            / f"exp_52_seed{seed}_multi_mutation_seed_results.csv"
        )
    )

seed_df = pd.concat(
    seed_frames,
    ignore_index=True,
).sort_values(
    "seed"
).reset_index(
    drop=True
)


# ============================================================
# 4. Aggregate fold results
# ============================================================

fold_frames = [
    pd.read_csv(
        BASE_DIR
        / "exp_51_multi_mutation_fold_results.csv"
    )
]

for seed in NEW_SEEDS:
    fold_frames.append(
        pd.read_csv(
            BASE_DIR
            / f"exp_52_seed{seed}_multi_mutation_fold_results.csv"
        )
    )

fold_df = pd.concat(
    fold_frames,
    ignore_index=True,
).sort_values(
    [
        "seed",
        "fold",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 5. Aggregate hard predictions
# ============================================================

prediction_frames = [
    pd.read_csv(
        BASE_DIR
        / "exp_51_multi_mutation_predictions.csv"
    )
]

for seed in NEW_SEEDS:
    prediction_frames.append(
        pd.read_csv(
            BASE_DIR
            / f"exp_52_seed{seed}_multi_mutation_predictions.csv"
        )
    )

prediction_df = pd.concat(
    prediction_frames,
    ignore_index=True,
).sort_values(
    [
        "seed",
        "row_index",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 6. Aggregate classwise results
#
# exp_51 classwise CSV에는 seed 열이 없으므로
# 여기서 명시적으로 추가한다.
# ============================================================

classwise_frames = []

seed7_classwise = pd.read_csv(
    BASE_DIR
    / "exp_51_multi_mutation_classwise_results.csv"
)

seed7_classwise.insert(
    0,
    "seed",
    7,
)

classwise_frames.append(
    seed7_classwise
)

for seed in NEW_SEEDS:

    frame = pd.read_csv(
        BASE_DIR
        / f"exp_52_seed{seed}_multi_mutation_classwise_results.csv"
    )

    frame.insert(
        0,
        "seed",
        seed,
    )

    classwise_frames.append(
        frame
    )

classwise_df = pd.concat(
    classwise_frames,
    ignore_index=True,
).sort_values(
    [
        "seed",
        "subclass",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 7. Aggregate 26-class OOF probabilities
#
# exp_51 probability CSV에도 seed 열이 없으므로
# 명시적으로 추가한다.
# ============================================================

probability_frames = []

seed7_prob = pd.read_csv(
    BASE_DIR
    / "exp_51_multi_mutation_oof_probabilities.csv"
)

seed7_prob.insert(
    0,
    "seed",
    7,
)

probability_frames.append(
    seed7_prob
)

for seed in NEW_SEEDS:

    frame = pd.read_csv(
        BASE_DIR
        / f"exp_52_seed{seed}_multi_mutation_oof_probabilities.csv"
    )

    frame.insert(
        0,
        "seed",
        seed,
    )

    probability_frames.append(
        frame
    )

probability_df = pd.concat(
    probability_frames,
    ignore_index=True,
).sort_values(
    [
        "seed",
        "row_index",
    ]
).reset_index(
    drop=True
)


# ============================================================
# 8. Save combined exp_52 results
# ============================================================

seed_df.to_csv(
    BASE_DIR
    / "exp_52_multi_mutation_seed_results.csv",
    index=False,
)

fold_df.to_csv(
    BASE_DIR
    / "exp_52_multi_mutation_fold_results.csv",
    index=False,
)

prediction_df.to_csv(
    BASE_DIR
    / "exp_52_multi_mutation_predictions.csv",
    index=False,
)

classwise_df.to_csv(
    BASE_DIR
    / "exp_52_multi_mutation_classwise_results.csv",
    index=False,
)

probability_df.to_csv(
    BASE_DIR
    / "exp_52_multi_mutation_oof_probabilities.csv",
    index=False,
)


# ============================================================
# 9. Repeated-CV summary
# ============================================================

all_baselines_exact = bool(
    (
        seed_df[
            "baseline_reproduction_rate"
        ].eq(1.0)
    )
    .all()
    and
    (
        seed_df[
            "baseline_mismatch_count"
        ].eq(0)
    )
    .all()
)

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
    "exp_52 REPEATED-CV SUMMARY"
)

print(
    "=" * 80
)

print(
    seed_df[
        [
            "seed",
            "baseline_xgb_oof_f1",
            "candidate_xgb_oof_f1",
            "xgb_diff",
            "baseline_80_20_oof_f1",
            "candidate_80_20_oof_f1",
            "ensemble_diff",
            "improved_folds",
            "candidate_rescue",
            "candidate_damage",
            "net_correct_gain",
            "baseline_reproduction_rate",
            "baseline_mismatch_count",
        ]
    ].to_string(
        index=False
    )
)

print()

print(
    "모든 baseline 100% 재현:",
    all_baselines_exact,
)

print(
    "개선 Seed:",
    f"{positive_seeds}/3",
)

print(
    "개선 Fold:",
    f"{positive_folds}/15",
)

print()

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

print(
    "해석 원칙:"
)

print(
    "- seed 7은 사전 screen 결과를 재사용"
)

print(
    "- seed 77/777은 새로운 독립 반복 검증"
)

print(
    "- Group-aware 검증은 아직 수행하지 않음"
)

print(
    "- BEST 승격이나 TEST 추론은 아직 하지 않음"
)

print(
    "\nTEST FILE: NEVER READ"
)

print(
    "=" * 80
)
