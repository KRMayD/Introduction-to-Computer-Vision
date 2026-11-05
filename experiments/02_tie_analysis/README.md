# 02. 동률 처리 영향 분석

Manhattan 거리와 기존 8개 k에 대해 같은 이웃을 사용하면서 동률 해결 규칙만 바꿉니다. 동률 비율, 오답→정답·정답→오답의 수와 CV 지표 변화를 분석합니다. 이 실험은 CV 전용입니다.

```bash
python experiments/02_tie_analysis/run.py --results runs/01_baseline --output runs/02_tie_analysis --device cpu
```

[tie_analysis.py](../../knn_cifar10/tie_analysis.py)가 기본 실험의 OOF 예측을 재현하는지 확인한 뒤 대안을 계산합니다. 기본 실험을 이동해 데이터 경로가 바뀌었다면 `--data`로 새 경로를 지정할 수 있습니다.

[저장된 결과](../../results/02_tie_analysis/README.md) · [전체 실험 안내](../README.md)
