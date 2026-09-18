# Stack7 CatBoost GPU Meta 54/162/541 실행 패키지

이 패키지는 검증된 기존 7개 모델 확률 bank와 동일한 V3 baseline을 사용해 CatBoost GPU meta learner만 새로 학습합니다. 기존 7개 모델은 재학습하지 않습니다. 고정 grid는 `depth [3,5,7] × iterations [200,600,1200] × l2_leaf_reg [3,30] × weighting [none,sqrt_inverse,balanced]`의 54 head, 3-fold 162 fit, 540 blend와 baseline control 1개로 총 541행입니다. 표현은 203열 `log_probability_confidence`로 고정됩니다.

새 fit 직전마다 CatBoost GPU0 접근성과 GPU0 free memory를 검사하며 CPU fit fallback은 없습니다. 완료 cache replay와 CPU `predict_proba(thread_count=4)`는 GPU admission을 요구하지 않습니다. 결과는 `CACHED_BASE_OOF_META_CV_ADAPTIVE_NOT_NESTED` 개발 증거이며 독립 nested 검증이 아닙니다. test/full OOF는 search 선택 동결 뒤 deployment에서만 읽습니다.

Windows WSL 실행 순서:

```text
cd /mnt/e/GPU-Workspace/cbj-gene-subtype/staged_stack7_catboost_20260916_v1
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B -m unittest discover -s tests -v
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B synthetic_gpu_smoke.py --out runs/smoke_catboost_v1
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py preflight --config runtime_config.windows.json --phase search
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py search --config runtime_config.windows.json --start
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py status --config runtime_config.windows.json
```

Search가 모두 완료되고 `FROZEN_SELECTION.json`이 검증된 뒤에만 사람이 deployment를 별도로 시작합니다.

```text
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py preflight --config runtime_config.windows.json --phase deploy
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py deploy --config runtime_config.windows.json --start
```

`search`/`deploy`는 명시적 `--start` 없이는 시작하지 않습니다. 재실행은 identity/hash가 일치하는 완료 cache만 재사용합니다. 다음 grid, cross-peer blend, website submission은 자동 실행하지 않습니다.
