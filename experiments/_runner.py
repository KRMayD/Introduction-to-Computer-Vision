"""Share CLI defaults without duplicating the KNN implementation."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(experiment, module, defaults=None):
    options = {"--output": str(ROOT / "runs" / experiment), **(defaults or {})}
    supplied = sys.argv[1:]
    prefix = []
    for flag, value in options.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in supplied):
            prefix.extend([flag, *([str(v) for v in value] if isinstance(value, (tuple, list)) else [str(value)])])
    sys.argv[1:1] = prefix
    importlib.import_module(module).main()
