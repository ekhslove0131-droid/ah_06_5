# Drive에서 받은 이전 최고점 재현 자료

개인 CBJ Drive의 `CBJ_BEST_0520406_SUBMISSION_REPLAY_20260913`에서 2026-09-16에 직접 내려받은 소스다. 팀원 폴더나 데이터 폴더를 통째로 복제한 자료가 아니다.

- [Colab 노트북](best05204/CBJ_BEST_0520406_FROZEN_REPLAY.ipynb): 저장된 모델로 이전 최고점 OOF와 제출 예측을 재생하는 코드
- [학습 코드 안내](best05204/README.md), `best05204/source/`: H1 mRMR+global+raw, C10 TF-IDF, fold 내부 bias, 기하평균·support guard에 필요한 원본 코드
- [환경 잠금 파일](best05204/requirements.lock.txt), [고정 설정](best05204/best05204_config.json)
- [과거 Colab 실행 영수증](best05204/COLAB_EXECUTION_RECEIPT_20260914.json): 내부 Macro-F1 **0.5204059148667912**, 저장 OOF와 예측 클래스 일치
- [다운로드 출처와 원본/공개본 해시](PROVENANCE.json)

## 재현 전 준비

노트북의 입력 설정을 먼저 확인하고, 본인에게 접근 권한이 있는 원래 Drive 모델 묶음과 대회 train/test/sample 파일을 준비해야 한다. 모델 묶음과 원본 행 데이터는 이 공개 저장소에 포함하지 않았다. Colab T4 환경을 사용하며, 노트북은 저장된 모델 재생용이지 새로운 모델 탐색용이 아니다.

하위 원본 README가 언급하는 `receipts/` 및 `audits/` 전체, 모델 ZIP은 이 공개본에 없다. 포함된 Colab 영수증은 과거 실행 기록이며 이번 다운로드에서 다시 학습하거나 재실행한 결과가 아니다. 현재 Windows 최고점 **0.546567**을 Colab에서 재현하는 노트북으로 해석하면 안 된다.

공개 노트북은 출력·실행 메타데이터를 제거했다. 모든 셀의 소스가 다운로드 원본과 동일하고 노트북 형식 검사가 통과했으며, 소스·설정·lock·영수증은 원본 byte 그대로다. 새 Colab top-to-bottom 실행은 **NOT VERIFIED**다.
