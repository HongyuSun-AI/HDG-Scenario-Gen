import hashlib
import numpy as np
from .congestion_lanechange_experiment import candidates
from .left_lanechange_experiment import lane_change
from .experiment_data import initialization
from .representation import decode
from .road import MergeRoad


def select_initializations(data, groups, seed):
    road = MergeRoad(data['metadata']['map_path'])
    items = candidates(data, road, seed)
    chosen, seen = [], set()
    for candidate in items:
        item = dict(candidate)
        _, first, _, ego, _, reference = initialization(data, item)
        states, valid, _ = decode(first)
        joint = states[valid]
        joint = joint[np.lexsort((joint[:, 1], joint[:, 0]))]
        key = hashlib.sha256(np.round(np.r_[states[ego], joint.ravel()], 3).tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        event = lane_change(dict(tracks=reference, ego=ego, collision_time=None,
                                 arrival_time=None, exit_time=None), road)
        item.update(group=len(chosen), seed=seed + 1000 * (len(chosen) + 1),
                    initial_state_fingerprint=key, intended_from_lane=2, intended_to_lane=1,
                    intent_source='Recorded lane 2→1 merge',
                    reference_lane_change=event)
        chosen.append(item)
        if len(chosen) == groups:
            break
    if len(chosen) != groups:
        raise ValueError('Only {} distinct; requested {}'.format(len(chosen), groups))
    return dict(groups=chosen, seed=seed, eligible_candidates=len(items), selection='Recorded merge intention')
