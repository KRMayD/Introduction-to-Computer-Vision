"""Run 04_k101; all distance and voting code is shared."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments._runner import run

if __name__ == "__main__":
    # 실제 흐름: _runner.run() → voting.main() → voting.run() → core.ExactKNN.
    # uniform_nearest는 다수결 + 최근접 동률 해결이며 거리 가중 투표와 구분된다.
    # max-k=101은 1,3,...,101의 홀수 51개를 의미한다. 기본 실험의 데이터와 fold를 재사용한다.
    # 사용자가 준 명령행 옵션은 _runner에서 우선 적용된다.
    run('04_k101', 'knn_cifar10.voting', {'--baseline': ROOT / "runs" / "01_baseline", '--voting-methods': ['uniform_nearest'], '--max-k': 101})
