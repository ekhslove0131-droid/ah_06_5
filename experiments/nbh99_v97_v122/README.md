# nbh99 v97 → v122 연구 코드

이 폴더는 최종 제출로 이어진 두 핵심 연구 스크립트와 검증 receipt를 보존한다.

- `advanced_stack_kaggle_v97_sealed_gene_router_ensemble.py`: CBJ 부모 위에 3-seed
  gene-aware router 합의를 적용한 v97
- `v97_decision.json`: v97 OOF 및 안정성 결과
- `advanced_stack_kaggle_v122_multires_complement_final.py`: 정확 변이 프로필과 정확
  유전자 집합의 상보 라벨 합의로 v97 한 건을 교정한 v122 연구 코드

## 성능 흐름

| 버전 | 내부 검증 | Public LB |
|---|---:|---:|
| CBJ 부모 | OOF `0.5970141422` | `0.4772888064` |
| v97 | OOF `0.6034121303` | `0.4907219947` |
| v122 | 구조 검증 정확도 `1.0000` | `0.4922388606` |

v122의 `1.0000`은 일반 OOF Macro F1이 아니라 train 상보쌍에 대한 구조적
leave-one-profile-side-out 정확도다. 일반 OOF와 직접 비교하지 않는다.

원본 스크립트는 대회 데이터와 로컬 OOF/test 확률 산출물에 의존한다. 저장소에는 원본
데이터나 대용량 확률 파일을 넣지 않았다. 최종 제출의 독립적인 byte 재현은
`final/nbh99_v122/reproduce_v122_from_v97.py`를 사용한다.
