# 04. k=101까지 확장

다수결 + 최근접 동률 해결 방식으로 세 거리와 `k=1,3,…,101`의 모든 홀수 51개를 비교합니다. 전체 153개 설정·765개 fold 결과를 생성합니다.

```bash
python experiments/04_k101/run.py --baseline runs/01_baseline --output runs/04_k101 --device cpu
```

진입점은 [voting.py](../../knn_cifar10/voting.py)에 기본 인자 `--voting-methods uniform_nearest --max-k 101`을 전달합니다. 모델 선택과 검증 조건은 기본 실험과 동일합니다. 거리 가중 투표의 k 확장은 이 실험에 포함하지 않았습니다.

[저장된 결과](../../results/04_k101/README.md) · [전체 실험 안내](../README.md)
