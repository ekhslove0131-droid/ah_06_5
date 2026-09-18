# 결과와 해석 범위

## 현재 최고 frozen lineage

| 구분 | 행 수 | Macro-F1 | 의미 |
|---|---:|---:|---|
| linear meta inner | 4,959 | 0.5273407417760553 | outer-0 train 영역 안의 3-fold meta 선택 자료 |
| CatBoost meta inner | 4,959 | 0.5273969154760471 | 같은 cached base OOF를 이용한 비선형 관점 |
| crossblend inner | 4,959 | **0.5303752528221235** | 202개 선언 혼합에서 선택된 값 |
| linear meta full adaptive OOF | 6,201 | 0.5459123490386504 | 선택 동결 후 보고용 전체 OOF |
| CatBoost meta full adaptive OOF | 6,201 | 0.5451693186036448 | 선택 동결 후 보고용 전체 OOF |
| crossblend full adaptive OOF | 6,201 | **0.5465670608677152** | 현재 최고 개발 기록 |
| crossblend 외부 평가 | 2,546 test | `NOT VERIFIED` | 현재 CSV의 평가서버 결과를 확인하지 않음 |

최종 crossblend는 linear 44%, nonlinear 56%의 산술평균이다. 선택된 선언 index는 112이며 추가 fit은 0회였다. 이전 linear 최고보다 full adaptive OOF가 `+0.0006547118290648468` 높았고 test argmax는 145행 달랐다.

## 결과 계보

| 단계 | full adaptive OOF | 주요 산출물 SHA-256 | 상태 |
|---|---:|---|---|
| combo V3 baseline | 0.5401371102785413 | predictions `33c0e538...916796` | 완료 receipt 확인 |
| linear Stack7 | 0.5459123490386504 | predictions `af024e7c...956547` | 완료 receipt·NumPy replay 확인 |
| CatBoost Stack7 | 0.5451693186036448 | predictions `b47a5306...2a55f` | 완료 receipt·native prediction audit 확인 |
| frozen crossblend | 0.5465670608677152 | predictions `0095317c...7910` | 완료 receipt·literal parent blend replay 확인 |

축약 SHA는 읽기 편의를 위한 표시다. exact hash는 `PRIVATE_INPUTS.json`과 frozen runtime config에 있다.

## 사용자 보고 외부 결과와의 분리

이전 linear Stack7 CSV SHA-256 `9fef8f5a3a118f81f3570ce5788f092403492e5b164832732fd57710d2ec70c8`에 대해 사용자가 외부 Macro-F1 `0.3925738796`을 보고했다. 이는 current crossblend가 아니라 이전 linear CSV의 결과이며, Codex가 평가 사이트에서 독립 확인한 값이 아니다. 외부 점수는 이후 학습·선택·가중치·보정 입력으로 사용하지 않았다.

current crossblend CSV SHA-256은 `5503812d0d10b11c5a9d45711e634ea630dbf5264259df10c43a19d6cdd41b93`이지만 CSV 자체는 공개 release에서 제외했고 외부 점수는 확인되지 않았다.

## 한계

- inner와 full 수치는 같은 7개 base OOF 계보를 재사용한 adaptive development evidence다.
- full 6,201행은 선택에 직접 쓰지 않았지만 완전한 nested whole-stack validation은 아니다.
- CatBoost 확인용 1,242행 `0.52685833`과 linear 확인용 1,242행 `0.5331664128227076`은 해당 meta fit에서 제외됐어도 과거에 관찰된 구간이다.
- public repository에는 data, OOF/probability arrays, models, cache, CSV가 없어 fresh-machine 결과 재생산을 증명하지 않는다.
- 목표 0.60/0.65/0.70은 달성하지 않았다.
