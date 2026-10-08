import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def activate():
    paths = [ROOT / 'third_party/nuplan-devkit', ROOT / 'third_party/tuplan_garage']


    if (ROOT / 'third_party/pdm_deps').is_dir():
        paths.insert(0, ROOT / 'third_party/pdm_deps')
    for path in reversed(paths):
        if not path.is_dir():
            raise RuntimeError(f'Missing PDM source directory: {path}')
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/hdg-mpl')
    os.environ.setdefault('XDG_CACHE_HOME', '/tmp/hdg-cache')
