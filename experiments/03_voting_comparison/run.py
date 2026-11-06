"""Run 03_voting_comparison; all distance and voting code is shared."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments._runner import run

if __name__ == "__main__":
    # voting.main()의 기본 세 투표 방식을 기존 8개 k·세 거리·같은 fold에서 비교한다.
    run('03_voting_comparison', 'knn_cifar10.voting', {'--baseline': ROOT / "runs" / "01_baseline"})
