# 03. 투표 방식 3종 비교

세 거리와 기존 8개 k에 대해 다음 세 투표 방식을 비교합니다.

- `uniform_class_id`: 다수결, 작은 클래스 번호로 동률 해결.
- `uniform_nearest`: 다수결, 동률인 최다 득표 클래스 중 가장 가까운 이웃의 클래스를 선택.
- `distance_weighted`: 실제 거리의 역수 가중 투표. 동률이면 최근접 후보 선택.

```bash
python experiments/03_voting_comparison/run.py --baseline runs/01_baseline --output runs/03_voting_comparison --device cpu
```

[voting.py](../../knn_cifar10/voting.py)를 사용합니다. 전체 72개 설정·360개 fold를 검증하고, 각 투표 방식과 거리의 CV 최적 k에 대해 총 9개 테스트 결과를 저장합니다. 기본 실험의 데이터 해시·fold·예측을 대조합니다.

거리 0인 이웃이 있으면 그 이웃들만 동일 가중치로 투표합니다. Euclidean 가중치는 제곱 거리가 아닌 실제 거리의 역수입니다.

[저장된 결과](../../results/03_voting_comparison/README.md) · [전체 실험 안내](../README.md)
