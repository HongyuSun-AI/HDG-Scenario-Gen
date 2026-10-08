import numpy as np
from .representation import velocities


def assess_risk(states, valid, road, dt=.1, reaction_time=1.5,
                max_deceleration=3.4, deviation=50., kappa=1., lengths=None):
    n, w = valid.shape
    lengths = np.full(n, 4.5) if lengths is None else np.asarray(lengths)
    speed = np.linalg.norm(velocities(states, valid, dt), axis=-1)
    lane = road.lane(states[..., 0], states[..., 1])
    indicators = np.zeros((n, n, w))
    severity = np.zeros_like(indicators)
    for t in range(w):
        for i in np.flatnonzero(valid[:, t] & (lane[:, t] >= 0)):
            current = lane[i, t]

            future = lane[i, t:min(w, t + max(2, round(1 / dt)))]
            changed = future[(future >= 0) & (future != current)]
            target = int(changed[0]) if len(changed) else current
            if current == 2 and road.bounds(min(states[i, t, 0] + 20, road.xmax))[2:].ptp() < 1.:
                target = 1
            neighbors = set()
            for ln in {current, target}:
                candidates = np.flatnonzero(valid[:, t] & (lane[:, t] == ln) & (np.arange(n) != i))
                if not len(candidates):
                    continue
                dx = states[candidates, t, 0] - states[i, t, 0]
                front = candidates[dx >= 0]
                rear = candidates[dx < 0]
                if len(front):
                    neighbors.add(int(front[np.argmin(states[front, t, 0])]))
                if ln != current and len(rear):
                    neighbors.add(int(rear[np.argmax(states[rear, t, 0])]))
            for j in neighbors:
                gap = road.gap(current, states[i, t, 0], states[j, t, 0], lengths[i], lengths[j])
                deficit = speed[i, t] * reaction_time + (speed[i, t] ** 2 - speed[j, t] ** 2) / (2 * max_deceleration) - gap
                indicators[i, j, t] = deficit > 0
                severity[i, j, t] = min(kappa * max(deficit, 0) / deviation, 1)

    counts = (valid[:, None] & valid[None]).sum(-1)
    exposure = indicators.sum(-1) / np.maximum(counts, 1)
    rs = 1 - np.prod(1 - exposure * severity.max(-1), axis=1)
    return rs.astype(np.float32)


def assess_risk_vectorized(states, valid, road, dt=.1, reaction_time=1.5,
                           max_deceleration=3.4, deviation=50., kappa=1., lengths=None,
                           return_components=False):

    n, w = valid.shape
    lengths = np.full(n, 4.5) if lengths is None else np.asarray(lengths)
    speed = np.linalg.norm(velocities(states, valid, dt), axis=-1)
    x = states[..., 0]
    lane = road.lane(x, states[..., 1])
    target = lane.copy()
    lookahead = max(2, round(1 / dt))

    for offset in reversed(range(1, min(lookahead, w))):
        changed = (lane[:, offset:] >= 0) & (lane[:, offset:] != lane[:, :-offset])
        target[:, :-offset] = np.where(changed, lane[:, offset:], target[:, :-offset])
    bounds = road.bounds(np.minimum(x + 20, road.xmax))
    target = np.where((lane == 2) & (np.ptp(bounds[2:], axis=0) < 1.), 1, target)
    eligible = valid[:, None, :] & (lane[:, None, :] >= 0) & valid[None, :, :]
    eligible &= ~np.eye(n, dtype=bool)[:, :, None]
    dx = x[None, :, :] - x[:, None, :]
    neighbors = np.zeros((n, n, w), dtype=bool)

    def add_nearest(candidate, front):

        value = np.where(candidate, x[None, :, :], np.inf if front else -np.inf)
        index = np.argmin(value, axis=1) if front else np.argmax(value, axis=1)
        chosen = np.zeros_like(candidate)
        np.put_along_axis(chosen, index[:, None, :], candidate.any(axis=1)[:, None, :], axis=1)
        return chosen

    current = eligible & (lane[None, :, :] == lane[:, None, :])
    changing = eligible & (lane[None, :, :] == target[:, None, :]) & (target[:, None, :] != lane[:, None, :])
    neighbors |= add_nearest(current & (dx >= 0), True)
    neighbors |= add_nearest(changing & (dx >= 0), True)
    neighbors |= add_nearest(changing & (dx < 0), False)
    gap = np.zeros((n, n, w))
    for ln in range(3):
        arc = road.path_s(ln, x)
        distances = abs(arc[:, None, :] - arc[None, :, :]) - (lengths[:, None, None] + lengths[None, :, None]) / 2
        gap = np.where(lane[:, None, :] == ln, distances, gap)
    deficit = (speed[:, None, :] * reaction_time +
               (speed[:, None, :] ** 2 - speed[None, :, :] ** 2) / (2 * max_deceleration) - gap)
    indicators = neighbors & (deficit > 0)


    peak_deficit = np.where(neighbors, np.maximum(deficit, 0) / deviation, 0).max(-1)
    counts = (valid[:, None, :] & valid[None, :, :]).sum(-1)
    exposure = indicators.sum(-1) / np.maximum(counts, 1)
    if return_components:
        return exposure, peak_deficit
    return risk_from_components(exposure, peak_deficit, kappa)


def risk_from_components(exposure, peak_deficit, kappa):

    return (1 - np.prod(1 - exposure * np.minimum(kappa * peak_deficit, 1), axis=-1)).astype(np.float32)
