# 실험별 결과

단위는 %입니다. 정밀도·재현율·F1은 Macro 평균이며, CV는 5개 fold의 평균 ± 표본 표준편차(ddof=1)입니다.

주실험은 전체 학습 50,000장에 Stratified 5-fold 검증을 적용했습니다. 최종 모델은 평균 CV 정확도로 선택한 뒤 전체 학습 데이터로 공식 테스트 10,000장을 평가했습니다.

| 실험 | 투표 방식 | 거리 | k | CV 정확도 | 테스트 정확도 | 테스트 정밀도 | 테스트 재현율 | 테스트 F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [01](01_baseline/README.md) | 다수결 + 작은 클래스 번호 | Manhattan | 1 | 37.08 ± 0.19 | 38.59 | 43.33 | 38.59 | 38.62 |
| [03](03_voting_comparison/README.md) | 다수결 + 최근접 동률 해결 | Manhattan | 5 | 37.85 ± 0.51 | 39.21 | 45.60 | 39.21 | 38.90 |
| [03](03_voting_comparison/README.md) | 거리 역수 가중 투표 | Manhattan | 5 | 38.19 ± 0.50 | 39.50 | 45.92 | 39.50 | 39.22 |
| [04](04_k101/README.md) | 다수결 + 최근접 동률 해결 | Manhattan | 5 | 37.85 ± 0.51 | 39.21 | 45.60 | 39.21 | 38.90 |

최고 관측 테스트 정확도는 **거리 역수 가중 투표, Manhattan, k=5의 39.50%**입니다. 최신 k 확장 실험은 **다수결 + 최근접 동률 해결, Manhattan, k=5의 39.21%**입니다. 각 모델의 거리와 k는 CV로 선택했습니다.

## 전체 표 바로가기

| 실험 | 설정 수 | fold 결과 수 | 결과 문서 |
| --- | --- | --- | --- |
| 01 기본 KNN | 24 | 120 | [결과](01_baseline/README.md) · [CV](01_baseline/cv_results.md) · [Fold](01_baseline/fold_results.md) · [테스트](01_baseline/test_results.md) |
| 02 동률 분석 | 8 | 40 | [결과](02_tie_analysis/README.md) · [동률](02_tie_analysis/tie_results.md) · [CV](02_tie_analysis/cv_results.md) · [Fold](02_tie_analysis/fold_results.md) |
| 03 투표 방식 비교 | 72 | 360 | [결과](03_voting_comparison/README.md) · [CV](03_voting_comparison/cv_results.md) · [Fold](03_voting_comparison/fold_results.md) · [테스트](03_voting_comparison/test_results.md) |
| 04 k 범위 확장 | 153 | 765 | [결과](04_k101/README.md) · [CV](04_k101/cv_results.md) · [Fold](04_k101/fold_results.md) · [테스트](04_k101/test_results.md) |

02는 Manhattan CV 전용 분석입니다. 03은 세 투표 방식의 결과를 포함하므로 01의 기본 투표 결과가 재현되어 있습니다. 위 설정 수는 실험별 기록 수이며, 중복을 제거한 고유 모델 수를 의미하지 않습니다.

## 결과 해석

- 기본 동률 규칙에서는 최적 k=1이었지만, 최근접 동률 해결과 거리 가중 투표에서는 최적 k=5였습니다. 같은 k라도 투표 규칙에 따라 결과가 달라집니다.
- k=101까지 확장해도 최근접 동률 해결 방식의 최적 설정은 바뀌지 않았습니다. k가 커질수록 정확도는 전반적으로 감소했으나 모든 인접 k에서 단조 감소한 것은 아닙니다.
- 원시 픽셀은 배경·위치·밝기의 영향을 받습니다. 큰 k에서 서로 다른 클래스가 섞이는 현상은 성능 감소의 가능한 해석이며, 원인을 직접 검증한 추가 실험은 수행하지 않았습니다.

## 파일과 재현

CSV는 원본 수치(0~1)를 보존하고, Markdown 표는 이를 백분율로 변환해 소수 둘째 자리까지 표시합니다. metadata.json은 최초 실행 기록이며 당시 로컬 경로도 보존했습니다. [원본 경로·코드 변경 내역](../docs/PROVENANCE.md)을 참고하세요.

저장된 결과에는 CSV·메타데이터·PNG를 포함했습니다. 데이터셋과 NPZ 이웃·예측 캐시는 업로드에서 제외했습니다. 새 실험은 [실행 안내](../README.md#실험-재현)에 따라 runs/에 생성하며, 02~04 실행 전 01을 실행해 predictions.npz를 생성해야 합니다.
