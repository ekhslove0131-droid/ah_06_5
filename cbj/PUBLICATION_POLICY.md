# 공개 범위 정책

## 포함

- 현재 최고 `0.5465670608677152`로 이어지는 6개 frozen source/config package
- package가 원래 결속한 source allowlist, manifest, fixture, test, acceptance metadata
- 공개 release verifier와 그 focused regression tests
- 결과 수치의 증거 등급과 재현 한계를 설명하는 문서
- 별도 provenance가 있는 sanitized Drive/Kaggle source snapshot

## 제외

- `train.csv`, `test.csv`, `sample_submission.csv`와 모든 row-level data
- OOF, probability, prediction NPZ와 제출 CSV
- joblib, CatBoost/model weight, cache, run directory와 binary output
- 실패하거나 사용하지 않은 과거 package version과 scratch/diff
- 중단된 launcher, agent brief/skill, credential, token, signed URL
- 실행되지 않은 `staged_crossblend_bias_20260916_v1` 및 newmax3 계열

## 주장 규칙

- `0.5465670608677152`는 full adaptive OOF 개발 기록으로만 쓴다.
- `0.5303752528221235`는 4,959행 inner 선택 점수이며 full 점수와 섞지 않는다.
- 이전 linear CSV의 사용자 보고 외부 `0.3925738796`은 current crossblend 성능으로 옮겨 적지 않는다.
- current crossblend 외부 점수, GitHub-only bit-exact replay, fresh-machine retraining, Colab 최신 재현은 `NOT VERIFIED`다.
- source/config snapshot 보존을 end-to-end 재현 완료로 표현하지 않는다.

## 변경 절차

1. 추가 파일이 이 정책의 포함 범위인지 사람이 검토한다.
2. notebook은 output과 execution count를 제거하고 비밀·row data가 없는지 확인한다.
3. trusted packaging 단계에서 `verify_release.py --write-manifest`를 실행한다.
4. 일반 verify를 다시 실행하고 diff/security review를 마친 뒤에만 공개한다.
