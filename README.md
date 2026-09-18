# Cancer Genomics Classification — BEST07 실험 공유

암종 26-class 분류 해커톤에서 진행한 Multi-Mutation Parser 기반 BEST07 실험을 공유합니다.

## 주요 폴더
- experiments/ : exp_51 ~ exp_54 실험 코드
- final/ : BEST07 최종 후보 코드
- docs/ : 실험 과정과 결과 정리

## 주요 결과
- BEST06 Public LB: 0.341068025
- BEST07 Public LB: 0.3440018301

자세한 실험 내용은 docs/BEST07_실험정리.md를 참고해주세요.

## 추가 공유: 안애영 exp66 (2026-09-18)

[exp66 코드·검증 결과·사용 안내](experiments/exp66/README.md)를 추가했습니다.
두 기본 모델의 확률을 nested 방식으로 결합한 TRAIN-only 연구 후보입니다.
3-seed 평균 OOF Macro F1은 **0.502438**이며, 같은 분할의 기존 Router **0.467093** 대비 **+0.035345**입니다.
**Public LB 점수가 아니며, Group-aware 검증과 최종 제출 추론은 미완료입니다.**
기존 BEST07 코드는 변경하지 않았습니다. 환자별 데이터와 OOF 확률은 저장소에 포함하지 않습니다.

## 추가 공유: nbh99 최종 v122 (2026-09-18)

[v97→v122 최종 코드·제출파일·검증 결과](final/nbh99_v122/README.md)를 추가했습니다.
최종 Public LB는 **0.4922388606**이며, 직전 v97 **0.4907219947** 대비
**+0.0015168659** 상승했습니다. 원본 대회 데이터와 대용량 확률 산출물은 포함하지
않고, 최종 제출 CSV 및 byte 단위 재현 스크립트를 제공합니다.
