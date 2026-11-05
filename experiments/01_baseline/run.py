"""Run 01_baseline; all distance and voting code is shared."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments._runner import run

if __name__ == "__main__":
    run('01_baseline', 'knn_cifar10.core', {})
