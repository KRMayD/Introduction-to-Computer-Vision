# CIFAR-10 KNN: 거리·투표 방식·k 비교

**한국어** | [English](README.en.md)

CIFAR-10 전체 학습 50,000장과 공식 테스트 10,000장으로 K-최근접 이웃(KNN)을 직접 구현하고 평가한 프로젝트입니다. 원본 RGB 픽셀 3,072차원에서 거리, 이웃 수 k, 투표 규칙을 비교했습니다.

거리 계산·이웃 탐색·투표는 직접 구현했습니다. scikit-learn은 데이터 분할, 평가 지표와 독립 검증에 사용했습니다. KNN의 학습 단계는 학습 데이터를 저장하는 과정입니다.

## 주요 결과

각 방식의 거리와 k는 **평균 5-fold CV 정확도**로 선택했습니다. 아래는 선택된 모델의 공식 테스트 결과입니다. 수치는 %이며 정밀도·재현율·F1은 Macro 평균입니다.

| 투표 방식 | 거리 | k | CV 정확도: 평균 ± 표준편차 | 테스트 정확도 | 정밀도 | 재현율 | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 다수결 + 작은 클래스 번호로 동률 해결 | Manhattan | 1 | 37.08 ± 0.19 | 38.59 | 43.33 | 38.59 | 38.62 |
| 다수결 + 최근접 이웃으로 동률 해결 | Manhattan | 5 | 37.85 ± 0.51 | 39.21 | 45.60 | 39.21 | 38.90 |
| 거리 역수 가중 투표 | Manhattan | 5 | 38.19 ± 0.50 | **39.50** | **45.92** | **39.50** | **39.22** |

최신 실험에서는 다수결 + 최근접 동률 해결의 k를 101까지 확장했습니다. 최적 설정은 **Manhattan, k=5**로 유지됐고 테스트 정확도도 **39.21%**로 같았습니다. 거리 가중 투표는 기존 8개 k에서 평가했습니다.

**[전체 결과와 비교 분석](results/README.md)** · **[평가 지표 설명](docs/METRICS.md)**

핵심 코드에는 한글 주석을 추가했습니다. 실행 파일의 연결 관계와 배열·거리·투표·검증 흐름은 **[코드 읽기 안내](docs/CODE_WALKTHROUGH.ko.md)**에서 확인할 수 있습니다.

## 실험 종류

기존 k 8개는 `1, 3, 5, 7, 9, 11, 15, 21`입니다.

| 번호 | 실험 | 거리 | k 범위 | CV 설정 수 | fold 결과 수 | 코드 / 결과 |
| --- | --- | --- | --- | --- | --- | --- |
| 01 | 기본 KNN: 다수결 + 작은 클래스 번호 | 3종 | 기존 8개 | 24 | 120 | [코드](experiments/01_baseline/run.py) / [결과](results/01_baseline/README.md) |
| 02 | 동률 발생 및 최근접 동률 해결 영향 | Manhattan | 기존 8개 | 8 | 40 | [코드](experiments/02_tie_analysis/run.py) / [결과](results/02_tie_analysis/README.md) |
| 03 | 세 투표 방식 비교 | 3종 | 기존 8개 | 72 | 360 | [코드](experiments/03_voting_comparison/run.py) / [결과](results/03_voting_comparison/README.md) |
| 04 | 다수결 + 최근접 동률 해결의 k 확장 | 3종 | 1, 3, …, 101: 51개 | 153 | 765 | [코드](experiments/04_k101/run.py) / [결과](results/04_k101/README.md) |

02는 CV 전용 진단이며 공식 테스트를 평가하지 않았습니다. 03에서는 비교 조건을 맞추기 위해 01의 기본 투표도 재실행해 기존 예측과 일치함을 확인했습니다. 설정 수는 실험별 기록 수입니다.

## 공통 실험 조건

| 항목 | 설정 |
| --- | --- |
| 데이터 | 학습 50,000장 전체, 테스트 10,000장 전체 |
| 특징 | 32 × 32 × 3 원본 uint8 픽셀을 펼친 3,072차원 벡터 |
| 전처리 | 데이터 축소·PCA·특징 추출·표준화 없이 원본 픽셀 사용 |
| 거리 | Euclidean(L2), Manhattan(L1), Cosine distance |
| 검증 | Stratified 5-fold, `shuffle=True`, `random_state=42` |
| fold 구성 | 학습 40,000장 / 검증 10,000장, 각 이미지는 한 번 검증 |
| 모델 선택 | 평균 CV 정확도 최대 → 작은 k → 거리 이름 순서 |
| 최종 평가 | 거리별 CV 최적 k를 선택하고 전체 학습 데이터로 공식 테스트 평가 |
| 지표 | Accuracy, Macro Precision, Macro Recall, Macro F1 |
| CV 표기 | fold 평균 ± 표본 표준편차(`ddof=1`), 신뢰구간 아님 |
| 최근접 탐색 | 전체 학습 데이터를 대상으로 정확 탐색, 쿼리 배치 128 |

테스트 점수는 각 모델의 거리와 k를 선택하는 데 사용하지 않았습니다. 과제 과정에서 테스트 결과를 반복해서 관찰했으므로, 여기서 관측된 최고 점수를 추가적인 독립 평가의 결과로 해석하지는 않습니다.

## 투표와 동률 처리

- **`uniform_class_id`**: 각 이웃이 1표를 갖습니다. 최다 득표 클래스가 동률이면 가장 작은 클래스 번호를 선택합니다.
- **`uniform_nearest`**: 각 이웃이 1표를 갖습니다. 최다 득표 클래스가 동률이면 그 클래스들에 속한 이웃 중 가장 가까운 이웃의 클래스를 선택합니다.
- **`distance_weighted`**: 각 이웃에 실제 거리의 역수 `1 / d`를 부여합니다. Euclidean은 제곱 거리 대신 실제 L2 거리를 사용합니다. 거리 0인 이웃이 있으면 그 이웃들만 동일한 가중치로 투표합니다. 가중치 합이 동률이면 최근접 동률 해결을 적용합니다(`rtol=1e-12`, `atol=0`).

이웃 간 거리가 같으면 안정 정렬을 통해 해당 fold의 학습 데이터 인덱스가 작은 이웃을 먼저 배치합니다. 홀수 k에서도 다중 클래스 투표는 `2:2:1`처럼 동률이 발생할 수 있습니다.

## 디렉토리 구조

```text
cifar10-knn/
├── README.md
├── README.en.md                     # English documentation
├── requirements.txt                 # CPU 실행·지표·그래프 의존성
├── requirements-gpu.txt             # 선택적 PyTorch 의존성
├── knn_cifar10/
│   ├── core.py                      # 공통 KNN 구현 + 기본 실험
│   ├── tie_analysis.py              # 동률 영향 분석
│   ├── voting.py                    # 세 투표 방식·k 확장 공통 실행기
│   └── reporting/                   # 기존 PNG·HTML 내보내기 코드
├── experiments/
│   ├── 01_baseline/                 # 기본 KNN 실행·설명
│   ├── 02_tie_analysis/             # 동률 분석 실행·설명
│   ├── 03_voting_comparison/        # 세 투표 방식 비교 실행·설명
│   └── 04_k101/                     # k=101 확장 실행·설명
├── results/
│   ├── README.md                    # 실험 간 비교
│   ├── comparison.csv              # 최적 모델 비교 원자료
│   ├── 01_baseline/                 # 결과 표·CSV·PNG·메타데이터
│   ├── 02_tie_analysis/
│   ├── 03_voting_comparison/
│   └── 04_k101/
├── scripts/
│   ├── build_result_tables.py       # 저장된 CSV → 전체 Markdown 표
│   └── validate_repository.py       # 수치·구성·문서 링크 검증
├── tests/test_knn.py                # 작은 예제로 독립 구현 검증
├── docs/
│   ├── METRICS.md                   # 지표 계산과 Macro 평균
│   ├── PROVENANCE.md                # 원본 → 정리된 경로 및 변경 내역
│   └── source_manifest.json         # 복사 파일 SHA-256
├── data/README.md                   # 데이터 준비 안내
└── runs/                            # 재실행 결과, Git 제외
```

## 실행 환경

실험 당시 환경은 Python **3.10.20**, NumPy **1.26.4**, scikit-learn **1.7.2**, PyTorch **2.6.0+cu124**였고, GPU 실행에는 단일 NVIDIA RTX A5000을 사용했습니다. 나머지 의존성의 버전은 [requirements.txt](requirements.txt)에 기록했습니다.

이미 의존성이 설치된 Python/Conda 환경에서 저장소 루트를 작업 디렉토리로 사용합니다. `--device cpu`로 CPU에서 실행할 수 있으며, PyTorch를 사용할 수 있는 환경에서는 `--device cuda:0`으로 바꿀 수 있습니다. `--device auto`는 보이는 CUDA GPU가 있으면 사용하고, 없으면 CPU를 사용합니다. CUDA 장치 번호는 `CUDA_VISIBLE_DEVICES`가 적용된 논리 번호입니다.

패키지가 필요한 환경에서는 사용자 환경에 의존성을 준비한 후 실행합니다. 이 저장소를 정리하는 과정에서는 패키지를 설치하거나 전체 데이터 실험을 다시 수행하지 않았습니다.

## 데이터 준비

데이터셋은 [CIFAR-10 공식 배포 페이지](https://www.cs.toronto.edu/~kriz/cifar.html)의 Python 버전을 사용합니다. 32×32 컬러 이미지 60,000장 중 학습 50,000장, 테스트 10,000장으로 구성됩니다. 데이터셋은 저장소에 포함하지 않습니다.

기본 경로는 `data/cifar-10-python.tar`입니다. 다른 위치에 있다면 `--data /path/to/cifar-10-python.tar`를 지정합니다. `.tar.gz` 경로도 사용할 수 있으며, 로더는 압축 형식을 자동 판별하고 아카이브 내부 파일을 메모리에서 읽습니다. 자세한 내용은 [데이터 준비 안내](data/README.md)를 참고하세요.

## 실험 재현

아래 명령은 **새 실험을 실행**합니다. 저장된 표만 확인하려면 [결과 문서](results/README.md)를 열면 됩니다. 모든 명령은 저장소 루트에서 실행합니다.

```bash
# 01. 기본 KNN: 전체 학습 데이터의 5-fold CV + 공식 테스트
python experiments/01_baseline/run.py --data data/cifar-10-python.tar --output runs/01_baseline --device cpu

# 02. Manhattan CV 예측으로 동률 처리의 영향 분석
python experiments/02_tie_analysis/run.py --results runs/01_baseline --output runs/02_tie_analysis --device cpu

# 03. 동일한 fold에서 투표 방식 세 가지 비교
python experiments/03_voting_comparison/run.py --baseline runs/01_baseline --output runs/03_voting_comparison --device cpu

# 04. 다수결 + 최근접 동률 해결, 홀수 k=1~101
python experiments/04_k101/run.py --baseline runs/01_baseline --output runs/04_k101 --device cpu
```

02~04는 01에서 생성한 `metadata.json`과 `predictions.npz`를 사용해 분할과 예측을 대조합니다. 업로드된 `results/01_baseline`은 CSV·PNG 위주의 결과 보관본이므로, 재실행 시에는 먼저 01을 실행하고 생성된 `runs/01_baseline`을 사용합니다.

출력 디렉토리가 이미 존재하면 중단합니다. 재실행할 때는 `--output`으로 새 경로를 지정하고, 이후 명령의 `--baseline` 또는 `--results`도 해당 경로로 바꿉니다. 데이터 위치가 바뀌었다면 02~04에도 `--data`를 지정할 수 있습니다. 03~04에서는 기존 데이터 해시와 일치하는지 확인합니다.

실험별 기본값과 세부 명령은 [experiments/README.md](experiments/README.md)에 정리했습니다.

## 표 생성과 검증

```bash
# 저장된 CSV에서 전체 결과 표 재생성: 모델 실행 없음
python scripts/build_result_tables.py

# 네 실험을 모두 새로 실행했다면 새 결과의 표 생성
python scripts/build_result_tables.py --results-root runs

# 수치·원본 복사·디렉토리·문서 링크 점검
python scripts/validate_repository.py

# 작은 예제로 거리·투표·평가 구현 검증
python -m unittest discover -s tests -v
```

CPU 환경에서는 GPU 전용 테스트가 건너뛰어집니다. 이 검증은 전체 CIFAR-10 실험을 재실행하지 않습니다.

정리 직후의 확인 결과는 [검증 기록](docs/VALIDATION.md)에 정리했습니다.

기존 이미지/HTML 내보내기 코드는 모듈로 실행할 수 있습니다. 새 실험에서 03·04의 그래프와 k 범위 비교도 생성하려면 다음 명령을 표 생성 전에 실행합니다. 저장된 GitHub 결과의 기본 진입점은 Markdown 문서입니다.

```bash
python -m knn_cifar10.reporting.voting_report runs/03_voting_comparison
python -m knn_cifar10.reporting.extended_report runs/04_k101 --previous runs/03_voting_comparison
```

## 결과 파일 읽는 법

| 파일 | 내용 |
| --- | --- |
| `cv_results.md` | 거리·k·투표 방식별 모든 CV 평균 ± 표준편차 |
| `fold_results.md` | 모든 fold의 개별 결과 |
| `test_results.md` | 거리별 CV 최적 k의 공식 테스트 결과 |
| `class_results.md` | 최종 모델의 클래스별 Precision·Recall·F1 |
| `confusion_matrices.md` | 실제 클래스 × 예측 클래스의 혼동행렬 |
| `comparison.md` | 투표 방식별 최적 모델 비교, 해당 실험에 포함 |
| `tie_results.md` | 동률 빈도·정확도 변화, 동률 분석 실험에 포함 |
| `range_comparison.md` | 기존 k 범위와 101까지 확장한 결과 비교 |
| `*.csv` | 원본 수치: 성능 지표는 0~1 단위 |
| `metadata.json` | 당시 설정·환경·모델 선택·실행 기록 |

## GitHub 업로드 범위

코드, README, Markdown 표, CSV, 메타데이터, PNG를 업로드하도록 구성했습니다. `.gitignore`는 데이터셋 아카이브, `runs/`, NPZ/NPY 캐시, ZIP, 로그와 환경·캐시 디렉토리를 제외합니다. 원본 실험 파일과 정리된 파일의 대응은 [PROVENANCE.md](docs/PROVENANCE.md)에 기록했습니다.

## 데이터셋 출처

Alex Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, 2009. 데이터셋 설명과 참고문헌은 [CIFAR 공식 페이지](https://www.cs.toronto.edu/~kriz/cifar.html)를 참고하세요.
