# 재현 범위와 실행 경계

## 1. Fresh clone에서 지원하는 경로: 공개 release audit

```bash
python3 -B cbj/tools/verify_release.py
```

지원 범위는 다음과 같다.

- 공개 release 전체 파일 inventory와 SHA-256 검증
- 5개 `SOURCE_MANIFEST.json`과 recovery package의 실제 63-file identity 검증 (`RECOVERY_ACCEPTANCE_20260916.json` 기준; `SOURCE_CLOSURE.json`은 이전 버전의 역사적 기록)
- 모든 Python 파일을 bytecode/`__pycache__` 생성 없이 parse/compile
- notebook이 있을 경우 cell output과 execution count가 제거됐는지 검증
- CSV, NPZ, joblib, model, cache, run output 같은 비공개·바이너리 artifact 부재 검증
- `PRIVATE_INPUTS.json`의 모든 항목을 `NOT_INCLUDED`로 표시

이 검사는 source archive의 무결성을 확인한다. 모델 성능, Windows/CUDA 동작, Kaggle/Colab 실행, 제출 CSV 생성 또는 외부 평가를 재현하지 않는다.

## 2. 기존 Windows workspace에서만 지원하는 경로

완료 당시 workspace와 `PRIVATE_INPUTS.json`에 적힌 exact hash artifact가 모두 남아 있을 때 package 자체의 read-only preflight/status를 사용할 수 있다. 대표 런타임은 다음이었다.

```text
/home/vital/.local/share/cbj-grid070-20260914-v1/runtime/bin/python
```

완료 계보의 package/config identity:

| package | source SHA-256 | runtime config SHA-256 |
|---|---|---|
| `staged_team_ensemble_20260916_recovery_v2` | `b56d607e5da1edf3f0e4c97e426182896f094ba35faf038735ceabe03b343b52` | `617fd8d1c3b256bf1468fa2387e8f907ce72dede31e551bffdb58c106f42a863` |
| `staged_combo_search_20260916_v3` | `96a6a4f5cd64f4437f6f39df1c87ef50cf50544160bfad3a2605bb42a7e354c5` | `c1f22d1d962bfa07c569c5580a386fe0eed12ff73d35fc24df843f6c76492df0` |
| `staged_stack7_20260916_v3` | `c2aef86f39912319515381d61a74d4c9d09c53a2c8821b05b4871b8069dcd206` | `f169ab6608a6bcd25060d75fd32567f69094a145882acdc0af4dcb02ca50c587` |
| `staged_stack7_meta_20260916_v1` | `e8b2ebb3d4a9db89e41594fbd7abbbb222de4a304f6e7a3342bd589be74d1626` | `628f3fbcc38716ff8ae521ca399de978a236369a7a12ebe1a758d01f22c664d1` |
| `staged_stack7_catboost_20260916_v1` | `ecd4637bdeccb33b65b07721203954b1cbdfcfcc287ed59f4dd904bdfef2b69f` | `b6875a9dbe9b72328632b9a462cabc8bbe5624373d59c1cd1458f2691b60e993` |
| `staged_stack7_crossblend_20260916_v1` | `006ea767ff9bb377d89a74d96fe5c90ac65dedaa2b4c0daba33277d3509cbc9b` | `05b8b297d4a3593d5bcb485fb9eeac9505bdb6492aca9e78e81f14dbc973efbc` |

각 package의 세부 preflight/status 명령은 package 안의 원본 README를 따른다. `--start`는 쓰기와 계산을 시작하는 명시적 gate다. 완료 자료를 단순 감사하려는 경우 `--start`를 붙이지 않는다.

## 3. 비공개 입력 계약

고정 데이터 identity는 다음과 같다.

| 입력 | shape | SHA-256 |
|---|---:|---|
| `train.csv` | 6,201 × 4,386 | `92418b8441d058cfc68e939dd88725610750be4bc8edc51253cffc72fc4fc0ab` |
| `test.csv` | 2,546 × 4,385 | `e7e7f29a9b6251308e470ae3fb040a6da0cd8fcb0adb87e67f7761631c6a1ef0` |
| `sample_submission.csv` | 2,546 × 2 | `1d0e9fe0b5ab5c763eab8c97130a06712e2ac2b428299481109b447d8f2b4d84` |
| `mrmr_global_oof.npz` | OOF probability 6,201 × 26 | `ab952812c5b779bbaee826bc9163579268cffbf50dd567fb2162c5600bec08e9` |
| `EVIDENCE.npz` | 124 × 4,959 × 26 candidate probability | `461c5930b0dd094f13c9659eb720676b624efac9ccceddf4f067f59659dd914f` |
| `TRAINING_BANK.npz` | inner 7 × 4,959 × 26; full 7 × 6,201 × 26 | `934119e7cf89d1bf0ee3405b192b288feb5444bfcf7bebff2ef8ee2612b4fb9f` |

이외에도 완료 replay는 원래 recovery cache, frozen selection/model artifact, parent export와 receipt를 요구한다. 이들은 공개 release에 포함되지 않았다. 일부 cache tree는 개별 receipt로만 결속돼 있고 공개 가능한 단일 directory digest가 없으므로, fresh clone에서 source부터 같은 결과를 만드는 절차는 `NOT VERIFIED`다.

## 4. 검증 등급

- `VERIFIED`: 공개 파일 해시와 source closure; 완료 당시 receipt에 결속된 결과 수치
- `AUDIT-ONLY`: fresh clone에서 실행 가능한 표준 라이브러리 검사
- `NOT INCLUDED`: raw data, probability arrays, saved models, cache/run outputs, 제출 CSV
- `NOT VERIFIED`: GitHub-only bit-exact replay, fresh-machine full retraining, Colab의 최신 0.546 결과, 현재 crossblend 외부 점수

## 5. Release manifest 갱신

파일을 추가한 뒤 자동 검증이 manifest를 임의로 승인하지는 않는다. 공개 범위를 사람이 먼저 검토한 trusted packaging 단계에서만 다음 명령으로 manifest를 재생성한다.

```bash
python3 -B cbj/tools/verify_release.py --write-manifest
python3 -B cbj/tools/verify_release.py
```

첫 명령도 금지 artifact, Python 구문, notebook output, source closure를 먼저 검사하며 실패 시 manifest를 쓰지 않는다.
