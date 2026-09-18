# CBJ 2-anchor team ensemble V4

이 번들은 외부·테스트 데이터를 사용하지 않고 `train.csv`와 고정 grouped outer folds만으로 128개 선언을 관리한다. 현재 실행 가능 항목은 124개이며, Windows 환경에서 GPU 빌드가 확인되지 않은 LightGBM 4개는 대체 모델 없이 `BLOCKED`로 남긴다.

## 고정 탐색 절차

- outer 5-fold는 seed43 고정 folds를 그대로 사용한다.
- 각 outer-train 안에서 grouped inner 3-fold를 만들고, A2 bias에는 다시 grouped sub-inner 3-fold OOF만 사용한다.
- A1과 A2는 서로 다른 anchor 축으로 Round 0 각각 140개씩, 합계 280개를 끝까지 평가한다.
- 승격은 inner OOF Macro-F1 0.55 이상, 모든 기존 승격 모델과 hard-label disagreement 0.03 이상, 최대 8개다.
- 메타 라운드는 현재 라운드의 모든 조합을 완료한 뒤에만 중단을 판단하며 최대 3회다.
- outer-valid 점수는 선택·가중치·bias·중단에 사용하지 않고, 5개 fold가 모두 끝난 뒤 한 번만 최종 OOF 점수를 계산한다.
- 0.60 이상이어도 현재 번들만 완료한다. 다음 번들 학습과 제출은 자동으로 하지 않는다.

## Windows에서 한 번만 실행

작업 폴더:

`/mnt/e/GPU-Workspace/cbj-gene-subtype/staged_team_ensemble_20260915_v1`

런타임:

`/mnt/e/GPU-Workspace/cbj-gene-subtype/grid070_environment_20260914_v1/run_grid070_env.sh`

1. native 합성 GPU 테스트:

```bash
/mnt/e/GPU-Workspace/cbj-gene-subtype/grid070_environment_20260914_v1/run_grid070_env.sh \
  python -W error -m unittest tests.test_gpu_native -v
```

2. 실제 데이터 무학습 preflight:

```bash
/mnt/e/GPU-Workspace/cbj-gene-subtype/grid070_environment_20260914_v1/run_grid070_env.sh \
  python cli.py preflight --config runtime_config.windows.json
```

3. 승인된 대규모 실행 시작:

```bash
/mnt/e/GPU-Workspace/cbj-gene-subtype/grid070_environment_20260914_v1/run_grid070_env.sh \
  python cli.py launch \
  --run-root /mnt/e/GPU-Workspace/cbj-gene-subtype/staged_team_ensemble_20260915_v1/runs/team_ensemble_v1 \
  --config runtime_config.windows.json \
  --start
```

프로세스는 분리 실행되고, 같은 run root에 두 번째 시작을 요청하면 `ALREADY_RUNNING`으로 거부된다. 중단 명령은 제공하지 않는다. 완료 여부는 사용자가 결과 확인을 요청했을 때 `status`로 읽는다.

## 결과 해석

- `COMPLETE.json`: 번들 종료 상태와 최종 OOF 점수
- `FINAL_PROCEDURE_OOF.npz`: 모든 canonical ID를 정확히 한 번 포함한 최종 procedure OOF
- `STATE.json`, `HEARTBEAT.json`, `events.jsonl`: 재시작과 사후 확인을 위한 내구 상태
- `TARGET_REACHED`: 최종 OOF가 0.60 이상
- `BUNDLE_COMPLETE_REVIEW`: 0.60 미만이지만 현재 번들은 정상 완료

두 상태 모두 자동 추가 학습과 제출을 금지한다.
