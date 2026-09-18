# Stack7 Meta 54/162/541 실행 패키지

이 패키지는 이미 준비된 7개 모델의 확률 bank만 사용해 의도적으로 CPU에서 meta learner를 실행합니다. 기존 7개 모델은 재학습하지 않습니다. 고정 grid는 head 54개, inner fit 162개, 혼합 540개와 baseline control 1개로 총 541행입니다. 결과는 `CACHED_BASE_OOF_META_CV_ADAPTIVE_NOT_NESTED` 증거이며 독립 nested 검증이 아닙니다.

필요 데이터와 V2/V3 receipt/cache는 Windows에 이미 존재해야 합니다. 사전검증은 source/config/manifest, bank, frozen V3 inner baseline, ID/group/fold/class 순서를 읽기만 하며 run directory를 만들지 않습니다. 모든 불일치는 fail-closed이고 기존 파일은 덮어쓰지 않습니다.

Windows WSL에서 패키지 디렉터리로 이동한 뒤 아래 명령을 그대로 사용합니다.

```text
cd /mnt/e/GPU-Workspace/cbj-gene-subtype/staged_stack7_meta_20260916_v1
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py preflight --config runtime_config.windows.json --phase search
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py search --config runtime_config.windows.json --start
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py status --config runtime_config.windows.json
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py preflight --config runtime_config.windows.json --phase deploy
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python -B cli.py deploy --config runtime_config.windows.json --start
```

동일한 search/deploy 명령을 다시 실행하면 검증된 완료 fit/score/output만 재사용합니다. 다른 worker가 활성화되어 있으면 시작을 거부합니다. Search와 deployment는 별도 run root이며 search 완료 후에만 사람이 deployment를 호출합니다. 다음 perspective/grid를 자동으로 열지 않으며 competition website 제출도 수행하지 않습니다.

로그의 `standalone`은 개별 meta head 진단값이고, `bestinner`는 4,959행 search 점수입니다(기존 기준 `0.521184765169131`). 이를 기존 전체 6,201행 점수 `0.5401371102785413`과 직접 비교하면 안 됩니다. 전체 점수와 CSV는 search 선택을 동결한 뒤 별도 deployment 평가에서만 생성합니다.

Mac 합성 검증 명령:

```text
cd /Users/baital/dev/finalhack/cbj_windows_lab/staged_stack7_meta_20260916_v1
/Users/baital/dev/finalhack/cbj_windows_lab/submission_best05204/.venv-replay/bin/python -B -m unittest discover -s tests -v
```
