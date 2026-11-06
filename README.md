# CIFAR-10 KNN

**한국어** | [English](README.en.md)

CIFAR-10 전체 학습 50,000장과 공식 테스트 10,000장으로 거리·k·투표 규칙을 비교합니다. 실행 코드는 한글 주석이 있는 두 파일로 정리했습니다.

| 파일 | 역할 |
| --- | --- |
| [knn.py](knn.py) | 데이터 로드 → 거리 계산 → 이웃 탐색 → 투표 → 평가 |
| [experiment.py](experiment.py) | 실험 선택 → 5-fold CV → 최적 설정 선택 → 공식 테스트 |
| [results/](results/README.md) | 기존 실험의 전체 결과 표·CSV·그래프 |

## 실험 설정

원본 RGB 픽셀을 펼친 3,072차원 벡터를 사용하며 데이터 축소·표준화·PCA를 적용하지 않습니다. 거리와 투표 규칙은 직접 구현하고, SciPy는 CPU L1 계산, scikit-learn은 데이터 분할과 지표 계산에 사용합니다.

- 거리: Euclidean(L2), Manhattan(L1), Cosine.
- 기본 k: `1, 3, 5, 7, 9, 11, 15, 21`.
- 검증: Stratified 5-fold, shuffle, seed 42. fold마다 학습 40,000장 / 검증 10,000장.
- 선택: 평균 CV 정확도 최대 → 작은 k → 거리 이름 순서. 테스트 정답은 선택에 사용하지 않습니다.
- 최종 평가: 거리·투표 방식별 최적 k를 선택하고 전체 학습 데이터로 공식 테스트를 평가합니다.
- 지표: Accuracy, Macro Precision, Macro Recall, Macro F1. CV는 fold 평균과 표본 표준편차(`ddof=1`)입니다.

| `--experiment` | 실험 범위 | 기존 결과 |
| --- | --- | --- |
| `baseline` | 3개 거리 × 기본 8개 k, 작은 클래스 번호로 동률 해결 | [01](results/01_baseline/README.md) |
| `ties` | Manhattan × 기본 8개 k, 두 다수결 동률 규칙 비교, CV 전용 | [02](results/02_tie_analysis/README.md) |
| `voting` | 3개 거리 × 기본 8개 k × 세 투표 방식 | [03](results/03_voting_comparison/README.md) |
| `k101` | 3개 거리 × 홀수 k=1~101, 최근접 동률 해결 | [04](results/04_k101/README.md) |

## 세 투표 방식

- `uniform_class_id`: 이웃당 1표. 동률이면 가장 작은 클래스 번호를 선택합니다.
- `uniform_nearest`: 이웃당 1표. 동률이면 최다 득표 클래스들 중 가장 가까운 이웃의 클래스를 선택합니다. 다중 분류에서는 홀수 k도 `2:2:1`처럼 동률이 생깁니다.
- `distance_weighted`: 실제 거리의 역수 `1/d`를 누적합니다. 거리 0인 이웃이 있으면 그 이웃들만 동일 가중치로 투표합니다. 가중치 동률(`rtol=1e-12`)은 최근접 후보로 해결합니다.

동일 거리의 이웃은 학습 인덱스 순서를 유지합니다. 배치 단위 정확 탐색이며 근사 탐색을 사용하지 않습니다.

## 실험 재현

Python 3.10 기준 의존성은 [requirements.txt](requirements.txt)에 있습니다. CPU 실행에는 PyTorch가 필요 없습니다. CUDA 실행에는 기존 환경의 CUDA 지원 PyTorch를 사용합니다.

CIFAR-10 Python 형식의 신뢰할 수 있는 `.tar` 또는 `.tar.gz` 파일 경로를 지정하세요. 데이터셋은 저장소에 포함하지 않습니다.

```bash
python experiment.py --experiment voting --data /path/to/cifar-10-python.tar --output runs/voting_01 --device cpu
```

다른 실험은 `--experiment baseline`, `ties`, `k101`로 선택합니다. 각 실험은 이전 실험의 캐시 없이 독립적으로 실행됩니다. GPU는 `--device cuda:0` 또는 `auto`로 선택하며 `CUDA_VISIBLE_DEVICES`를 따릅니다. 전체 데이터의 정확 탐색이므로 CPU에서는 오래 걸릴 수 있습니다. 기본 배치 크기는 128, CPU 스레드는 4이며 옵션은 `python experiment.py --help`로 확인합니다.

출력: `cv_folds.csv`, `cv_summary.csv`, `selected_models.csv`, `test_results.csv`. `ties`는 선택·테스트 대신 `tie_folds.csv`, `tie_summary.csv`를 추가합니다. 점수는 0~1 범위이며 기존 출력 디렉토리는 덮어쓰지 않습니다. 그래프·보고서·캐시 생성은 실행부에서 제외했습니다.

## 기존 결과

단위는 %이며 Precision·Recall·F1은 Macro 평균입니다. 각 투표 방식의 거리와 k는 평균 CV 정확도로 선택했습니다.

| 투표 방식 | 거리 | k | CV 정확도 | 테스트 정확도 | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 작은 클래스 번호 동률 해결 | Manhattan | 1 | 37.08 ± 0.19 | 38.59 | 43.33 | 38.59 | 38.62 |
| 최근접 동률 해결 | Manhattan | 5 | 37.85 ± 0.51 | 39.21 | 45.60 | 39.21 | 38.90 |
| 거리 가중 투표 | Manhattan | 5 | 38.19 ± 0.50 | 39.50 | 45.92 | 39.50 | 39.22 |

최근접 동률 해결은 k=101까지 확장해도 최적 k=5가 유지됐습니다. [전체 표와 그래프](results/README.md)는 기존 실행에서 얻은 결과이며 이번 코드 정리 과정에서 전체 실험을 다시 돌리지는 않았습니다. 이전 코드와 문서는 Git 기록에 보존되어 있습니다.

정리 후 작은 데이터에서 네 실행 흐름을 검증하고, 저장된 이웃으로 재계산한 237개 검증·테스트 설정의 예측 11,370,000건이 기존 예측과 일치함을 확인했습니다. 실제 GPU 실행은 이번 검증에 포함하지 않았습니다.
