# nbh99 최종 제출 — v122

## 최종 결과

- Public Leaderboard Macro F1: **0.4922388606**
- 이전 최고 v97: **0.4907219947**
- 실제 상승: **+0.0015168659**
- 최종 제출: `submission_v122_final_lb_0.4922388606.csv`
- 제출 SHA256: `4733e9daa2f7ea5da108229009f3d654cd25a362936e8944924746cac9a3734c`

v122는 v97의 2,546개 예측 중 `TEST_1293` 한 건만 `GBMLGG → LGG`로
교정했다. 정확 변이 문자열과 정확 유전자 집합이라는 두 해상도에서 같은 상보 라벨
규칙이 독립적으로 재현됐을 때만 변경했다.

## 파일

- `submission_v122_final_lb_0.4922388606.csv`: 최종 제출 파일
- `submission_v97_parent.csv`: 직전 부모 제출 파일
- `reproduce_v122_from_v97.py`: 부모 CSV에서 최종 CSV를 byte 단위로 재현
- `../../experiments/nbh99_v97_v122/advanced_stack_kaggle_v122_multires_complement_final.py`:
  원래 workspace의 train-only 구조 검증 및 생성 코드(연구 참고용)
- `v122_decision.json`: 검증 결과와 생성 receipt

## 빠른 재현

Python 표준 라이브러리만으로 최종 파일을 재현할 수 있다.

```bash
python final/nbh99_v122/reproduce_v122_from_v97.py
```

출력 SHA256이 위 값과 같아야 한다. 전체 학습 재현에는 대회 원본 데이터와 저장소에
포함하지 않은 OOF/test 확률 산출물과 원래 workspace 경로 구성이 필요하다. 해당 연구
코드는 감사용으로 보존했으며, 원본 데이터는 대회 규정과 용량 문제로 커밋하지 않았다.

## 검증 근거

- v97 일반 OOF Macro F1: `0.6034121303`
- 변이 문자열 상보쌍: 443그룹, 구조 검증 886행, 정확도 `1.0000`
- 유전자 집합 상보쌍: 431그룹, 구조 검증 858행, 정확도 `1.0000`
- 두 해상도 최종 예측 불일치: 0건
- 최소 상보 매핑 support: 173
- 최소 상보 매핑 purity: `1.0000`

전체 실험 흐름과 탈락 후보는 `docs/nbh99/V97_V122_VALIDATION_REPORT.md`에 정리했다.
