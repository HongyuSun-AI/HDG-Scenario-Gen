from pathlib import Path
import sys
import numpy as np
import torch
from .representation import decode, sort_by_duration
from .risk import assess_risk_vectorized
from .road import MergeRoad


def prepare(source, destination, horizon=140, limit=512, seed=7, map_path='data_process/maps/merge.osm',
            kappa=1., binary_risk=False):


    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model_2'))
    try:
        old = torch.load(source, map_location='cpu', weights_only=False)
    finally:
        sys.path.pop(0)
    ids = np.random.default_rng(seed).choice(len(old), min(limit or len(old), len(old)), replace=False)
    road = MergeRoad(map_path)
    tracks, layouts, risks, individual, orders, kept = [], [], [], [], [], []
    skipped, grid_conflicts = 0, 0
    for idx in ids:
        x, order = sort_by_duration(old[int(idx)][0].float())
        x = x[..., :horizon].clone()
        raw = x.permute(1, 2, 0).numpy()
        physical, valid, malformed = decode(raw)

        if malformed.any() or ((raw != -1) & ((raw < 0) | (raw > 1))).any() or not valid[:, 0].any():
            skipped += 1
            continue
        layout, conflicts = road.layout(physical[:, 0], valid[:, 0])
        score = assess_risk_vectorized(physical, valid, road, kappa=kappa)
        tracks.append(x)
        layouts.append(torch.from_numpy(layout))
        individual.append(torch.from_numpy(score))
        risks.append(float(score.max()))
        orders.append(order)
        kept.append(int(idx))
        grid_conflicts += conflicts
    if len(tracks) < 2:
        raise ValueError('Insufficient clean scenarios')
    result = dict(tracks=torch.stack(tracks), initial=torch.stack(layouts),
                  risk=torch.tensor(risks), agent_risk=torch.stack(individual),
                  source_indices=torch.tensor(kept), order=torch.stack(orders),
                  metadata=dict(source=str(source), source_count=len(old), horizon=horizon, dt=.1,
                                map_path=map_path, seed=seed, skipped_unclean=skipped,
                                grid_conflicts=grid_conflicts, bounds=[[1000, 935, -3.14], [1140, 955, 3.14]],
                                split='Recording IDs unavailable; training/demo',
                                vehicle_length=4.5, vehicle_width=1.8,
                                risk_parameters=dict(reaction_time=1.5, max_deceleration=3.4, deviation=50., kappa=kappa)))
    if binary_risk:
        result['continuous_risk'] = result['risk'].clone()
        result['risk'] = (result['continuous_risk'] >= .9).float()
        result['metadata']['label_rule'] = 'risk>=0.9:1; risk<0.9:0'
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, destination)
    return result


def load(path):
    data = torch.load(path, map_location='cpu', weights_only=True)
    return data
