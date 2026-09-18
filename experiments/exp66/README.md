# 안애영 exp66 — 두 모델의 확률 결합, TRAIN-only nested 검증

팀원 연구 참고용입니다. 새 제출 모델이나 최종 대회 제출 스크립트가 아닙니다.

## 핵심 관점

XGBoost 80% + NLP 20% 고정 결합을, 두 모델이 출력한 26-class 확률(총 52개)을 입력받는 `StandardScaler + LogisticRegression(C=1, class_weight='balanced')`로 바꿨습니다. 기본 모델의 multi-mutation parser, depth3 XGBoost, inverse-sqrt class weight, 기존 NLP 설정은 유지했습니다. 새로운 specialist나 Router는 넣지 않았습니다.

각 outer 학습 구간 안에서 inner 3-fold로 기본 모델을 새로 학습하여 meta 학습용 OOF를 만들었습니다. outer 검증 환자는 meta 학습용 확률 생성과 signature 학습에서 제외합니다. 기존 exp56 확률은 동일 outer 검증 환자의 입력과 기준선 비교에만 사용합니다. 저장된 전체 OOF를 다시 나누어 meta 학습하는 방식이 아닙니다.

## 실제 결과

| Seed | 기존 기본 조합 | 기존 최종 Router | exp66 | Router 대비 |
|---|---:|---:|---:|---:|
| 19 | 0.439048 | 0.465061 | 0.502433 | +0.037372 |
| 193 | 0.438182 | 0.465585 | 0.501574 | +0.035989 |
| 2029 | 0.443222 | 0.470631 | 0.503306 | +0.032675 |
| 평균 | 0.440151 | 0.467093 | 0.502438 | +0.035345 |

- seed별 전체 OOF Macro F1을 먼저 계산한 뒤 세 값을 평균했습니다.
- 기본 조합 대비 15/15 fold 개선, 기존 Router 대비 14/15 fold 개선입니다.
- 과거 exp59의 3-seed 다수결 0.480366은 다른 집계 방식이므로 직접 비교하지 않습니다.
- 사전 screen 기준: 기본 조합 대비 +0.005 이상이며 4/5 fold 개선. 세 seed 모두 통과했습니다.
- 실행 시간 합계 약 26.3분. 이번 GitHub 공유를 위해 재학습하지 않았습니다.

**Public LB가 아닙니다. exp66의 Public 점수는 없으며, 기존 BEST를 대체하지 않았습니다.**

## 주의점과 팀 활용법

기존 Router 대비 26개 암종 중 19개가 평균적으로 개선됐지만, **DLBC(38명)는 평균 F1 0.48110 → 0.34309**로 세 seed 모두 하락했습니다. THYM, PCPG, OV도 세 seed 모두 하락했습니다. 따라서 희귀 암종 보완 모델이라고 단정하지 마세요.

팀 모델과 결합할 가치는 단독 점수보다 서로 다른 환자의 오류를 고치는지에 달려 있습니다. TRAIN OOF의 ID·클래스 순서뿐 아니라 outer/inner split과 group 정의, 학습 이력을 확인해야 합니다. 서로 다른 검증 체계의 OOF를 단순 결합한 점수는 확정적인 개선 증거가 아닙니다.

현재 **Group-aware exp66, 팀 모델과의 실제 결합 검증, 최종 전체 학습 및 TEST 추론은 미완료**입니다. 같은 seed들이 기존 실험에서도 사용됐으므로 완전히 봉인된 평가 결과라고 주장하지 않습니다. 기존 NLP의 내부 보정 방법도 유지했으며, TF-IDF는 기본 모델 학습 구간에서 fit하지만 그 내부 보정 fold마다 별도로 다시 fit한 구조는 아닙니다.

## 포함 파일

- `exp_66_nested_probability_stacker.py`: 실제 실행한 원본 코드.
- `exp_63_nlp_event_gene_screen.py`, `exp_65_model_capacity_screen.py`: exp66이 import하는 원본 helper. **두 helper 파일 자체를 실행하지 마세요.** main은 별도의 과거 실험이며, import 시 실행되지 않습니다. exp66에서는 원래 NLP/depth3 helper만 사용합니다.
- `results/seed_results.csv`, `fold_results.csv`, `classwise_mean.csv`, `summary.json`: 환자별 정보가 없는 집계 결과.
- `environment.txt`: 실제 실행 환경 기록.

세 원본 Python 파일은 수정 없이 복사했습니다. `results/summary.json`의 source SHA-256은 exp66 원본을 가리킵니다.

## 재실행 준비물 — GitHub 코드만으로는 실행되지 않습니다

팀 내부 공유 패키지에서 다음 참조 입력을 받아 이 폴더에 두어야 합니다. 원본 데이터는 대회에서 허용된 방식으로 준비하세요.

1. `train.csv`
2. 선택한 seed의 `exp_56_seed{seed}_class_weight_oof_probabilities.csv`
3. 선택한 seed의 `exp_56_seed{seed}_class_weight_predictions.csv`

지원 seed는 `19`, `193`, `2029`입니다. 참조 입력은 실행할 seed와 정확히 일치해야 합니다. Python 환경에 필요한 패키지는 `environment.txt`를 참고하세요. 다른 플랫폼/버전에서의 재현은 별도 확인이 필요합니다.

프로젝트 루트에서 seed19 실행 예시:

```sh
python experiments/exp66/exp_66_nested_probability_stacker.py --seed 19
```

출력은 스크립트 옆 `exp_66_nested_stacker_seed19/`에 생성됩니다. 이 폴더가 이미 있으면 의도적으로 중단합니다. 기존 결과를 지우거나 덮어쓰지 말고 새 작업 사본에서 실행하세요. 기존 확률 재현 assert가 실패하면 원인을 확인해야 하며, assert를 삭제해서 통과시키지 마세요.

## 데이터와 규칙

이번 exp66은 TEST를 읽지 않았고 TEST 통계·예측 빈도·행 수·리더보드 피드백을 학습/선택에 사용하지 않았습니다. 외부 데이터나 팀원 코드를 복사하지 않았습니다. 이 설명은 이번 실험의 범위이며, 전체 대회 적격성을 대신 보증하지는 않습니다.

GitHub에는 원본 TRAIN/TEST, 환자별 정답·OOF 확률·예측, 모델 바이너리를 올리지 않았습니다. 이를 필요로 하는 팀원은 내부 공유 패키지를 이용해 주세요. 이 코드는 `/data` 경로를 포함한 대회 최종 단일 제출 파일이 아닙니다.
