import numpy as np
from scipy.optimize import lsq_linear
from .representation import LOW, HIGH, segments

CONFIG = dict(version='vehicle_envelope_v1', length=4.5, width=1.8,
              clearance_m=.05, second_difference_weight=4., iterations=4,
              scope='Future-only, fixed history/padding',
              constraints='Full-body road boundaries',
              correction_limit_m=None)


def center_limits(x, heading, road):

    dx = np.array([2.25, 2.25, -2.25, -2.25])
    dy = np.array([.9, -.9, -.9, .9])
    c, s = np.cos(heading)[:, None], np.sin(heading)[:, None]
    ox, oy = c*dx-s*dy, s*dx+c*dy
    b = road.bounds(x[:, None]+ox)
    lo = (b[-1]-oy).max(axis=1)+CONFIG['clearance_m']
    hi = (b[0]-oy).min(axis=1)-CONFIG['clearance_m']

    return np.maximum(lo, LOW[1]), np.minimum(hi, HIGH[1])


def _project_future_to_road(tracks, cut, road, representable_heading=False):
    out = np.array(tracks, copy=True)
    valid = np.isfinite(tracks).all(0) & (tracks >= -.1).all(0)
    audit = dict(repaired_segments=0, corrected_points=0, max_lateral_correction_m=0.,
                 solver_fallbacks=0, heading_fallbacks=0, remaining_violations=0)
    for agent in range(tracks.shape[1]):
        for a, b in segments(valid[agent]):
            start = max(a, cut)
            if start >= b: continue
            physical = np.clip(tracks[:, agent, a:b].T, 0, 1)*(HIGH-LOW)+LOW
            k = start-a
            future = physical[k:].copy()
            x, target, heading = future.T.copy()
            lo, hi = center_limits(x, heading, road)
            if ((target >= lo) & (target <= hi)).all(): continue

            history = physical[max(0, k-2):k]
            h = len(history); n = len(target)
            D = np.diff(np.eye(n+h), n=2, axis=0)
            A = np.vstack([np.eye(n), CONFIG['second_difference_weight']*D[:, h:]])
            rhs = np.r_[target, -CONFIG['second_difference_weight']*D[:, :h]@history[:, 1]]
            y = target.copy()
            for iteration in range(CONFIG['iterations']):
                if representable_heading:


                    encoded=((np.clip(heading, LOW[2], HIGH[2])-LOW[2])/(HIGH[2]-LOW[2])).astype(out.dtype)
                    heading=encoded.astype(np.float64)*(HIGH[2]-LOW[2])+LOW[2]
                lo, hi = center_limits(x, heading, road)
                impossible = lo >= hi
                if impossible.any():

                    slope = (road.center(1, x+.25)-road.center(1, x-.25))/.5
                    heading[impossible] = np.arctan(slope[impossible])
                    audit['heading_fallbacks'] += int(impossible.sum())
                    lo, hi = center_limits(x, heading, road)
                if (lo >= hi).any():
                    raise ValueError('Vehicle exceeds road envelope')
                fit = lsq_linear(A, rhs, bounds=(lo, hi), method='bvls', tol=1e-7, max_iter=100)
                if fit.success and np.isfinite(fit.x).all(): y = fit.x
                else:
                    audit['solver_fallbacks'] += 1
                    y = np.clip(target, lo, hi)
                if iteration == CONFIG['iterations']-1: break
                xy = np.column_stack([x, y])
                delta = np.diff(np.vstack([history[-1, :2], xy]), axis=0) if h else np.diff(xy, axis=0)
                if h:
                    moving = np.linalg.norm(delta, axis=1) > .06
                    heading[moving] = np.arctan2(delta[moving, 1], delta[moving, 0])
                elif n > 1:
                    delta = np.vstack([delta[0], delta])
                    moving = np.linalg.norm(delta, axis=1) > .06
                    heading[moving] = np.arctan2(delta[moving, 1], delta[moving, 0])

            out[1, agent, start:b] = (y-LOW[1])/(HIGH[1]-LOW[1])
            out[2, agent, start:b] = (heading-LOW[2])/(HIGH[2]-LOW[2])
            shift = np.abs(y-target)
            audit['repaired_segments'] += 1
            audit['corrected_points'] += int((shift > 1e-6).sum())
            audit['max_lateral_correction_m'] = max(audit['max_lateral_correction_m'], float(shift.max()))
            stored = np.clip(out[:, agent, start:b].T, 0, 1)*(HIGH-LOW)+LOW
            lower, upper = center_limits(stored[:, 0], stored[:, 2], road)
            audit['remaining_violations'] += int(((stored[:, 1] < lower-1e-4) | (stored[:, 1] > upper+1e-4)).sum())
    if audit['remaining_violations']: raise ValueError('Generated road projection failed validation')
    return out, audit


def project_future_to_road(tracks, cut, road):


    try:
        return _project_future_to_road(tracks, cut, road)
    except ValueError as exc:
        if str(exc) != "Generated road projection failed validation":
            raise
        out, audit = _project_future_to_road(tracks, cut, road, representable_heading=True)
        audit['representable_heading_retry'] = True
        return out, audit
