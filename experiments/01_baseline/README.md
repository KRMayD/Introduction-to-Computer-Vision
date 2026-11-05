# 01. 기본 KNN

Euclidean·Manhattan·Cosine 거리와 `k=1,3,5,7,9,11,15,21`의 조합 24개를 평가합니다. 각 이웃은 1표를 갖고, 투표 동률이면 작은 클래스 번호를 선택합니다.

```bash
python experiments/01_baseline/run.py --data data/cifar-10-python.tar --output runs/01_baseline --device cpu
```

공통 구현은 [core.py](../../knn_cifar10/core.py)입니다. 전체 학습 데이터에 Stratified 5-fold CV를 수행한 뒤 거리별 최적 k를 공식 테스트에 평가합니다. `predictions.npz`는 후속 실험이 기존 예측과 분할을 대조하는 데 사용합니다.

[저장된 결과](../../results/01_baseline/README.md) · [전체 실험 안내](../README.md)
