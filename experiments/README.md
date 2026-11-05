# 실험 실행 코드

모든 실험은 [공통 KNN 구현](../knn_cifar10/core.py)을 사용합니다. 각 `run.py`는 해당 실험의 기본 인자를 설정합니다. 명령은 저장소 루트에서 실행하며, `--help`로 사용 가능한 옵션을 확인할 수 있습니다.

| 실험 | 진입점 | 기본 출력 | 필요 입력 |
| --- | --- | --- | --- |
| 기본 KNN | [01_baseline/run.py](01_baseline/run.py) | `runs/01_baseline` | CIFAR-10 Python 아카이브 |
| 동률 분석 | [02_tie_analysis/run.py](02_tie_analysis/run.py) | `runs/02_tie_analysis` | 기본 실험 결과와 데이터 |
| 투표 방식 비교 | [03_voting_comparison/run.py](03_voting_comparison/run.py) | `runs/03_voting_comparison` | 기본 실험 결과와 데이터 |
| k=101 확장 | [04_k101/run.py](04_k101/run.py) | `runs/04_k101` | 기본 실험 결과와 데이터 |

01을 먼저 실행한 다음 02~04를 실행합니다. 기본 실험을 다른 경로에 생성했다면 02의 `--results`, 03~04의 `--baseline`으로 해당 경로를 지정합니다. 출력 디렉토리가 이미 있으면 중단하므로, 반복 실행은 `--output`으로 새 경로를 지정합니다.

실험별 목적과 명령은 각 디렉토리의 README에 있습니다. 전체 재현 명령은 [상위 README](../README.md#실험-재현), 저장된 결과는 [결과 목록](../results/README.md)을 참고하세요.
