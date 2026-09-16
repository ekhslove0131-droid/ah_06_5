# Kaggle 완료 소스 스냅샷

2026-09-16에 개인 계정 `bjcoding`의 노트북 4개와 코드 전용 dataset의 의존 파일 4개를 직접 내려받았다. 내려받을 때 네 노트북 모두 원격 상태가 `COMPLETE`였다. 이는 과거 실행 완료 상태이며 모든 개별 후보의 성공이나 이번 저장소에서의 재현을 뜻하지 않는다.

| 폴더 | 역할 |
|---|---|
| `cbj-subclass-full-eda-and-xgb-baseline/` | 초기 데이터 관찰, 변이 표현, XGBoost 기준 모델 |
| `cbj-subclass-anchor-preparation/` | 공통 anchor와 평가에 필요한 자료 준비 |
| `cbj-subclass-anchor-evaluation/` | anchor 평가·결합 관점 확인 |
| `cbj-catboost-t4x2/` | 앞선 준비·평가 결과를 사용하는 CatBoost GPU 실험 |
| `shared_code/` | CatBoost 실행에 연결된 코드 전용 dataset의 Python 3개와 설정 1개 |

이 자료는 현재 Windows 결합 결과까지의 배경과 재현 의존성을 보존한다. 최신 0.546567 모델이 이 Kaggle 노트북에서 생성됐다는 뜻은 아니다. 실패 실행의 출력·로그나 폐기된 별도 버전은 수집하지 않았다. 원본 소스의 공통 함수와 선언된 후보를 임의로 삭제하지는 않았다.

## 실행 조건

각 `kernel-metadata.json`의 `dataset_sources`, `kernel_sources`, `docker_image`, GPU 설정을 먼저 확인한다. 비공개 `bjcoding/cbj-subclass-mutation-input-private` 데이터와 연결된 선행 노트북 출력은 공개본에 없고 별도 접근 권한이 필요하다. 특히 CatBoost 노트북은 preparation/evaluation 선행 출력 및 원래 `/kaggle/input/` 배치를 요구한다. `shared_code/`는 코드 복사본이며 Kaggle dataset mount를 자동으로 만들지 않는다.

`is_private: true`는 원래 Kaggle 노트북의 설정이다. Kaggle 원본을 공개 전환하거나 새 버전을 게시·실행하지 않았다. 여기에는 사용자가 승인한 GitHub 소스 복사본만 공개한다.

## 이번에 확인한 범위

- [PROVENANCE.json](PROVENANCE.json): 출처 URL, 다운로드 원본과 공개본 SHA-256
- 노트북 출력·실행 메타데이터 제거, 셀 소스 동일성 및 `nbformat` 형식 검사 통과
- Python/config/metadata 파일은 다운로드 원본과 byte 동일
- 데이터·모델·OOF·제출 CSV·인증정보 미포함
- 새 Kaggle/로컬 top-to-bottom 실행 및 최신 Windows 결과의 Kaggle 재현: **NOT VERIFIED**
