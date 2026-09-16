# Stack7 frozen peer crossblend

이 패키지는 동결된 linear meta winner와 CatBoost meta winner의 확률만 합성합니다. base/meta fit, bias fit, threshold 조정, 새 후보 탐색은 없습니다. INNER search는 alpha `0.00..1.00` 101개와 arithmetic/geometric 두 pooling의 202 선언행만 평가하며, 양 endpoint의 geometric 행은 arithmetic 행의 명시적 alias입니다.

세 단계 모두 쓰기에는 `--start`가 필요합니다. Parent export는 각 parent 패키지를 독립 subprocess에서 검증하므로 Python module 이름 충돌을 피합니다.

```text
cd /mnt/e/GPU-Workspace/cbj-gene-subtype/staged_stack7_crossblend_20260916_v1
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B -m unittest discover -s tests -v
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B crossblend.py export-inner --config runtime_config.windows.json --start
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B crossblend.py search --config runtime_config.windows.json --start
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B crossblend.py deploy --config runtime_config.windows.json --start
```

`export-inner`는 양 parent search가 동결된 뒤, `search`는 두 INNER export가 완전할 때, `deploy`는 cross selection과 양 parent deployment가 완전할 때만 진행됩니다. Search는 full/test를 읽지 않습니다. Deployment의 full OOF score는 보고 전용이고 선택에 쓰이지 않습니다. Competition website 제출은 수행하지 않습니다.
