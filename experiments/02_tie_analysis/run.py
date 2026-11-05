"""Run 02_tie_analysis; all distance and voting code is shared."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments._runner import run

if __name__ == "__main__":
    run('02_tie_analysis', 'knn_cifar10.tie_analysis', {'--results': ROOT / "runs" / "01_baseline"})
