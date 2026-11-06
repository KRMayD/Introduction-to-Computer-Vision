# 원본 파일과 정리된 구조

기존 작업 디렉토리의 코드·데이터·결과는 보존하고, 별도 `cifar10-knn` 디렉토리에 GitHub 업로드용 사본을 구성했습니다. 원본 전체 데이터 실험은 2026-11-04에 수행했습니다. 이 정리 과정에서는 데이터셋을 복사하거나 전체 실험을 다시 실행하지 않았습니다.

## 코드 대응

| 기존 파일 | 정리된 파일 | 역할 |
| --- | --- | --- |
| `knn_cifar10.py` | [knn_cifar10/core.py](../knn_cifar10/core.py) | 거리·탐색·투표·지표·기본 실험 |
| `analyze_voting_ties.py` | [knn_cifar10/tie_analysis.py](../knn_cifar10/tie_analysis.py) | Manhattan CV 동률 분석 |
| `knn_weighted_experiment.py` | [knn_cifar10/voting.py](../knn_cifar10/voting.py) | 투표 방식 비교와 k 확장 |
| `export_result_images.py` | [reporting/baseline_images.py](../knn_cifar10/reporting/baseline_images.py) | 기존 기본 결과 이미지 내보내기 |
| `export_weighted_results.py` | [reporting/voting_report.py](../knn_cifar10/reporting/voting_report.py) | 투표 비교 PNG·HTML 내보내기 |
| `export_extended_results.py` | [reporting/extended_report.py](../knn_cifar10/reporting/extended_report.py) | k 확장 PNG·HTML 내보내기 |
| `test_knn_cifar10.py` | [tests/test_knn.py](../tests/test_knn.py) | 독립 예제 검증 |

새 실험별 `run.py`는 공통 모듈을 호출하고 기본 입력·출력 경로를 설정하는 진입점입니다. 네 실험마다 KNN 구현을 복제하지 않았습니다.

## 결과 대응

| 기존 `results/` 하위 디렉토리 | 정리된 디렉토리 |
| --- | --- |
| `full_20261104_150743` | [01_baseline](../results/01_baseline/README.md) |
| `voting_analysis_20261104_151920` | [02_tie_analysis](../results/02_tie_analysis/README.md) |
| `weighted_full_20261104_154210` | [03_voting_comparison](../results/03_voting_comparison/README.md) |
| `nearest_k101_20261104_160407` | [04_k101](../results/04_k101/README.md) |

CSV·metadata.json·PNG **50개 파일**을 바이트 단위로 동일하게 복사했습니다. 기존 NPZ 예측·이웃 캐시, ZIP, 실행 로그, matplotlib 캐시와 중복 HTML 보고서는 원래 위치에 보존했습니다. GitHub에서는 새 Markdown 표를 통해 결과를 읽도록 구성했습니다.

초기 실행이 완료되지 않아 로그만 남은 디렉토리는 완료된 실험 결과에 포함하지 않았습니다. 외부 연구 비교 이미지는 이번 네 가지 실험 결과에 포함하지 않았습니다.

## 정리 과정의 코드 변경 범위

- 공통 구현을 Python 패키지로 옮기고 import 경로를 조정했습니다.
- 기본 데이터 경로를 저장소의 `data/`로 변경하고, 새 실험 결과는 `runs/`에 생성하도록 진입점을 추가했습니다.
- 후속 실험에 `--data`를 추가해 기본 실험을 이동했을 때 데이터 경로를 지정할 수 있게 했습니다. 투표 비교에서는 기본 실험의 해시와 대조합니다.
- 동률 분석에 `--output`을 추가하고, 후속 실행의 기본 장치를 `auto`로 설정했습니다.
- 동률 분석이 4개 CV 지표 요약도 저장하도록 결과 출력을 보완했습니다.
- 실행 명령 기록은 실제 진입점의 인자를 기록하도록 조정했습니다.

거리 공식·정확 탐색·투표 규칙·CV 분할·모델 선택·평가 지표의 계산은 기존 구현을 유지했습니다. 새 Markdown 표는 CSV에서 백분율을 계산해 만들며, 원본 수치를 변경하지 않습니다.

## 해시와 검증

2026-11-06에 핵심 구현과 실험 진입점에 한글 설명 주석을 추가했습니다. 수정 전후 8개 Python 파일의 AST를 비교해 실행 코드가 동일함을 확인했습니다. 원본 파일 해시는 유지하고, 수정된 공통 코드의 기존 사본 해시는 `organized_sha256_before_korean_comments`에 보관했습니다. `organized_sha256`은 현재 주석이 포함된 파일의 해시입니다. [코드 읽기 안내](CODE_WALKTHROUGH.ko.md)에 실행 흐름을 설명했습니다.

[source_manifest.json](source_manifest.json)에 원본·정리된 상대 경로와 SHA-256을 기록했습니다. `python scripts/validate_repository.py`는 정리된 파일의 해시, CV 요약과 fold 값의 일치, 테스트 지표와 혼동행렬의 일치, 결과 표 행 수와 문서 링크를 확인합니다.

metadata.json은 원본 실행 기록을 그대로 보존하므로 당시 머신의 로컬 경로가 포함됩니다. 재현할 때는 상위 README의 새 실행 명령과 자신의 데이터 경로를 사용합니다.
