"""Share CLI defaults without duplicating the KNN implementation."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def run(experiment, module, defaults=None):
    # 각 실험은 설정만 다르고 실제 구현은 공유한다. 기본 출력은 runs/<실험 이름>이다.
    options = {"--output": str(ROOT / "runs" / experiment), **(defaults or {})}
    supplied = sys.argv[1:]
    prefix = []
    for flag, value in options.items():
        # 사용자가 직접 준 옵션은 유지하고, 생략한 옵션만 기본값으로 채운다.
        # --output path와 --output=path 형태를 모두 인식한다.
        if not any(arg == flag or arg.startswith(flag + "=") for arg in supplied):
            prefix.extend([flag, *([str(v) for v in value] if isinstance(value, (tuple, list)) else [str(value)])])
    # 공통 모듈의 argparse가 같은 옵션을 읽도록 기본값을 명령행 인자 앞에 붙인다.
    sys.argv[1:1] = prefix
    # 모듈 이름으로 불러온 뒤 main()을 호출한다. 여기서는 거리·투표 계산을 하지 않는다.
    importlib.import_module(module).main()
