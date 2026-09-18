# CBJ best05204 training source and evidence

이 폴더는 최고점 제출 후보가 어떤 학습 코드와 검증 결과에서 만들어졌는지 Drive 화면에서 직접 확인하기 위한 보존본이다.

## 실제 실행 위치

- 모델 학습은 Windows의 RTX 3070 Ti와 CPU에서 수행됐다.
- Drive의 `CBJ_BEST_0520406_FROZEN_REPLAY.ipynb`는 저장 모델 OOF 재현과 test 추론을 T4에서 실행했다. 새 학습은 하지 않았다.
- 따라서 Colab 출력은 재현 흔적이고, 이 폴더의 Python 파일과 완료 영수증이 원 학습 흔적이다.

## 학습 단계와 코드

1. `source/cbj_lab.py`, `source/mcmp_repr.py`, `source/cbj_summary_view.py`: grouped fold, MCMP 표현, fold-local selection, XGBoost 공통 학습 기반.
2. `source/goal055_h1_runner.py`: mRMR+global29+raw12 H1 XGBoost 5-fold 학습.
3. `source/goal055_tfidf_logistic.py`: fold-local TF-IDF+numeric41 C10 로지스틱 5-fold 후보 학습.
4. `source/goal055_nested_macro_bias_v2.py`: 각 outer fold 안에서 3개 inner C10 모델을 새로 학습하고 inner OOF만으로 클래스 bias를 산출.
5. `source/goal055_geometric_support_guard.py`, `source/goal055_support_guard_v2.py`: H1/C10 확률 결합과 입력 기반 fallback. 이 단계는 새 분류기를 학습하지 않는다.
6. `best05204_config.json`: 제출에 사용한 최종 고정 파이프라인 계약.

## 완료 영수증

- `receipts/H1_COMPLETE.json`: H1 GPU fit 5회.
- `receipts/TFIDF_LOGISTIC_COMPLETE.json`: C1/C10 CPU fit 10회.
- `receipts/NESTED_MACRO_BIAS_COMPLETE.json`: nested inner CPU fit 15회와 모델/변환 경로.
- `audits/GOAL055_NESTED_C10_BIAS_AUDIT.json`: 최종 grouped OOF Macro-F1 0.5204059148667912 검증.
- `COLAB_EXECUTION_RECEIPT_20260914.json`: Colab T4에서 OOF/CSV 재현 완료.

학습·선택·bias 산출에는 test 또는 평가 서버 결과를 사용하지 않았다. 최종 test 파일은 파이프라인 고정 후 제출 예측 생성에만 읽었다.
