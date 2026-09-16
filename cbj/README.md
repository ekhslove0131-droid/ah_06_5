# CBJ Gene SUBCLASS frozen release

이 폴더는 2026-09-16에 완료된 CBJ Gene `SUBCLASS` 분류 계보의 공개 가능한 소스·설정 스냅샷이다. 현재 최고 기록은 crossblend의 전체 adaptive OOF Macro-F1 `0.5465670608677152`다. 이 값은 저장된 base OOF를 재사용한 개발 결과이며 독립 nested 검증이나 외부 평가 점수가 아니다.

## 먼저 확인할 것

```bash
python3 -B cbj/tools/verify_release.py
```

이 명령은 release 파일 해시, 각 Windows package의 원래 source closure, Python 구문, notebook output 제거 여부와 금지된 데이터·모델 파일 부재를 검사한다. ML 라이브러리나 비공개 데이터는 필요하지 않다.

## 구성

- `windows/`: 성공 계보에 직접 연결된 6개 frozen package의 원본 source/config/test 스냅샷
- `docs/results.md`: inner, full adaptive OOF, 사용자 보고 외부 결과를 구분한 결과표
- `REPRODUCTION.md`: 공개 저장소 audit와 기존 Windows workspace replay의 경계
- `PRIVATE_INPUTS.json`: 공개하지 않은 입력·cache·예측물의 파일명, 역할, SHA-256, shape
- `PUBLICATION_POLICY.md`: 포함·제외 기준과 주장 범위
- [drive/](drive/README.md): 이전 최고점 0.5204059의 Colab 재현 노트북, 학습 소스와 과거 실행 기록
- [kaggle/](kaggle/README.md): 완료된 개인 Kaggle 노트북 4개와 코드 전용 의존 파일
- `RELEASE_MANIFEST.json`: trusted packaging 단계에서 생성한 공개 파일 전체 해시

## 성공 계보

```text
staged_team_ensemble_20260916_recovery_v2
  -> staged_combo_search_20260916_v3
  -> staged_stack7_20260916_v3
     -> staged_stack7_meta_20260916_v1
     -> staged_stack7_catboost_20260916_v1
        -> staged_stack7_crossblend_20260916_v1
```

각 디렉터리의 `runtime_config.windows.json`은 완료 당시 Windows 경로와 해시를 보존한 역사적 스냅샷이다. 절대 경로를 새 환경에 맞게 조용히 바꾸지 않았다. 이 저장소만으로 fresh-machine bit-exact 재학습·재배포가 된다는 뜻은 아니다.

## 현재 결과 요약

- 검색용 inner 4,959행: `0.5303752528221235`
- 전체 adaptive OOF 6,201행: `0.5465670608677152`
- 선택: linear 44%, nonlinear 56%, arithmetic blend
- 현재 crossblend 외부 평가: `NOT VERIFIED`
- 학습/제출 CSV, raw data, OOF/NPZ, model/cache는 공개하지 않음

세부 수치와 해석 제한은 [docs/results.md](docs/results.md), 실행 조건은 [REPRODUCTION.md](REPRODUCTION.md)를 따른다.
