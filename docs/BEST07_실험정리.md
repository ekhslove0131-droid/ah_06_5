# BEST07 실험 정리

## 1. 실험 목적

기존 mutation parser가 한 유전자 셀에 여러 mutation이 들어 있는 경우에도
mutation type과 position을 한 번만 추출하고 있다는 점을 확인했습니다.

예:
- `Q369* I368N`
- `S622S G827R`
- `Y435fs N193fs`

TRAIN 데이터 기준:

- non-WT cell: 218,893
- multi-mutation cell: 22,026
- multi-mutation 비율: 약 10.06%
- 실제 whitespace mutation token: 255,164
- 기존 cell 단위 표현보다 추가로 확인된 token: 36,271

NLP 문서는 기존에도 전체 mutation 문자열을 사용하고 있었기 때문에,
정보가 완전히 사라진 것은 아닙니다.

이번 실험의 핵심 가설은 다음과 같습니다.

> XGB의 mutation type / position 표현을 cell 단위가 아니라
> whitespace-separated mutation token 단위로 계산하면 성능이 개선되는가?

---

## 2. exp_51 — Multi-Mutation Parser 1차 Screen

변경 사항:

- binary feature 동일
- mutation_count 동일
- signature 동일
- NLP 동일
- XGB 설정 동일
- 80% XGB + 20% NLP 동일
- mutation type / position parser만 token 단위로 변경

Seed 7 결과:

| 지표 | Baseline | Multi-parser | 차이 |
|---|---:|---:|---:|
| XGB OOF Macro F1 | 0.418072 | 0.423400 | +0.005329 |
| 80:20 OOF Macro F1 | 0.423408 | 0.426511 | +0.003103 |

- 개선 Fold: 4/5
- Rescue: 144
- Damage: 120
- 정답 순증가: +24
- 기존 baseline prediction 재현율: 100%

사전에 정한 screen 조건을 통과했습니다.

---

## 3. exp_52 — 3 Seeds 반복 검증

Seeds:
- 7
- 77
- 777

| Seed | Baseline | Multi-parser | 차이 |
|---:|---:|---:|---:|
| 7 | 0.423408 | 0.426511 | +0.003103 |
| 77 | 0.434335 | 0.437370 | +0.003035 |
| 777 | 0.423569 | 0.427986 | +0.004417 |

평균:

- Baseline: 0.427104
- Candidate: 0.430622
- 평균 차이: +0.003518
- 개선 Seed: 3/3
- 개선 Fold: 11/15

주의:
동일 seeds를 여러 아이디어의 선택 과정에서 사용했기 때문에
3/3 개선을 완전히 독립적인 세 번의 검증으로 해석하지는 않습니다.

---

## 4. exp_53 — Group-aware 검증

Exact mutation profile이 동일한 환자가 TRAIN과 VALID에 동시에 들어가는 것을 막기 위해
StratifiedGroupKFold를 사용했습니다.

동일하게 맞춘 조건:

- Outer CV: StratifiedGroupKFold
- Inner signature CV: StratifiedGroupKFold
- NLP calibration CV: StratifiedGroupKFold
- 동일 row
- 동일 fold
- 동일 ID
- 동일 정답
- 동일 NLP prediction

Fair comparison audit: PASS

| Seed | Baseline | Multi-parser | 차이 |
|---:|---:|---:|---:|
| 7 | 0.417243 | 0.423931 | +0.006688 |
| 77 | 0.408089 | 0.416085 | +0.007996 |
| 777 | 0.417908 | 0.421496 | +0.003588 |

평균:

- Baseline: 0.414413
- Candidate: 0.420504
- 평균 차이: +0.006091
- 개선 Seed: 3/3
- 개선 Fold: 12/15

해석:
parser 기반 80:20 모델의 개선은 Group-aware 조건에서도 유지되었습니다.

주의:
최종 BEST07 전체 구조
(Multi-parser + Specialists + Router)의 Group-aware 검증은 별도로 수행하지 않았습니다.

---

## 5. exp_54 — Multi-parser + 기존 Two Specialists

기존 BEST05 구조:

기존 base
→ LGG ↔ GBMLGG Specialist
→ KIPAN ↔ KIRC Specialist

새 후보:

Multi-parser base
→ 동일 LGG ↔ GBMLGG Specialist
→ 동일 KIPAN ↔ KIRC Specialist

| Seed | 기존 BEST05 | 새 구조 | 차이 |
|---:|---:|---:|---:|
| 7 | 0.452903 | 0.456035 | +0.003132 |
| 77 | 0.458198 | 0.463249 | +0.005051 |
| 777 | 0.449344 | 0.453778 | +0.004434 |

기존 BEST05 평균 약 0.45348
→ 새 구조 평균 약 0.45769

평균 약 +0.00421 개선

---

## 6. BEST06 Router 적용

기존 BEST06 Frozen Router 규칙은 그대로 유지했습니다.

1. BEST05 = KIRC, exp32 = KIPAN → exp32 선택
2. BEST05 = GBMLGG, exp32 = LGG → exp32 선택
3. BEST05 = LGG, exp32 = GBMLGG → exp32 선택
4. 그 외 → BEST05 유지

기존 BEST06과 새 후보 비교:

| Seed | BEST06 | BEST07 Candidate | 차이 |
|---:|---:|---:|---:|
| 7 | 0.457211 | 0.460785 | +0.003575 |
| 77 | 0.463941 | 0.468315 | +0.004374 |
| 777 | 0.456082 | 0.460252 | +0.004170 |

평균:

- 기존 BEST06: 0.459078
- 새 후보: 0.463117
- 평균 차이: +0.004040
- 개선 Seed: 3/3
- 개선 Fold: 11/15

---

## 7. Classwise 결과

BEST06 → BEST07 Candidate 비교에서:

- 26개 클래스 중 18개 F1 개선
- 8개 F1 하락
- Rescue: 373
- Damage: 317
- 정답 순증가: +56

상대적으로 큰 개선:

- BLCA
- PCPG
- LUSC
- OV
- PRAD

상대적으로 하락한 클래스:

- CESC
- SKCM
- THYM
- HNSC
- COAD

기존 specialist 핵심 영역인
KIRC / KIPAN / LGG / GBMLGG는 최종 구조에서 대체로 유지되었습니다.

---

## 8. Public Leaderboard 확인

기존 BEST06:

- Public LB: 0.341068025

BEST07 확인용 제출:

- Public LB: 0.3440018301

차이:

- 약 +0.002934

TRAIN에서 확인한 개선 방향이 Public LB에서도 같은 방향으로 나타났습니다.

단, Public LB 결과를 보고 추가적인 class별 보정이나 parser 수정은 수행하지 않았습니다.

---

## 9. 현재 BEST07 Candidate 구조

Multi-Mutation Parser
→ XGB 80% + NLP 20%
→ LGG ↔ GBMLGG Specialist
→ KIPAN ↔ KIRC Specialist
→ 기존 Frozen Router

최종 제출용 코드는:

`final/BEST_07_multi_mutation_router_candidate_submit.py`

입니다.

---

## 10. 해석 시 주의사항

1. parser 기반 80:20 모델의 Group-aware 개선은 확인했습니다.
2. specialist와 router까지 포함한 최종 BEST07 전체 구조의 Group-aware 검증은 아직 별도로 수행하지 않았습니다.
3. 동일 seeds를 여러 실험 선택 과정에 사용했기 때문에 3/3 seed 개선을 완전히 독립적인 3회의 검증이라고 보지는 않습니다.
4. TEST 데이터는 모델 선택 과정에는 사용하지 않았으며, 후보 구조를 동결한 뒤 최종 inference 목적으로 사용했습니다.
5. TEST 예측 분포나 Public LB를 근거로 모델 구조를 추가 변경하지 않았습니다.
