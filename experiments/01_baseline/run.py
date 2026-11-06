"""Run 01_baseline; all distance and voting code is shared."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# 어느 작업 디렉토리에서 실행해도 저장소의 공통 패키지를 찾을 수 있게 한다.
sys.path.insert(0, str(ROOT))
from experiments._runner import run

if __name__ == "__main__":
    # 기본 8개 k와 세 거리를 core.main()으로 실행한다. 출력 기본값은 runs/01_baseline이다.
    run('01_baseline', 'knn_cifar10.core', {})
