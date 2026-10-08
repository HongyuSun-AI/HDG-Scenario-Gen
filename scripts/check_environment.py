import importlib.metadata as metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from hdg.pdm_planner import PDMClosedAdapter
from hdg.pdm_replay import FixedFutureTrajectory
from hdg.road import MergeRoad
from hdg.collision_demo import run

expected = {'torch': '2.3.1', 'numpy': '1.24.3',
            'scipy': '1.9.1', 'shapely': '2.0.7'}
actual = {name: metadata.version(name) for name in expected}
for name, version in actual.items():
    if version.split('+')[0] != expected[name]:
        raise RuntimeError('{}: expected {}, got {}'.format(name, expected[name], version))
road = MergeRoad(str(ROOT / 'data_process/maps/merge.osm'))
print(json.dumps({'python': sys.version, 'packages': actual,
                  'cuda_available': torch.cuda.is_available(),
                  'map_x_range': [road.xmin, road.xmax], 'pdm_import': 'ok'}, indent=2))
