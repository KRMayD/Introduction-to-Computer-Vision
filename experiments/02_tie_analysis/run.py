"""Run 02_tie_analysis; all distance and voting code is shared."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments._runner import run

if __name__ == "__main__":
    # 01의 OOF 예측을 기준으로 Manhattan 투표 동률을 분석한다. 공식 테스트는 평가하지 않는다.
    run('02_tie_analysis', 'knn_cifar10.tie_analysis', {'--results': ROOT / "runs" / "01_baseline"})
