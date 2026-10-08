import numpy as np
from .representation import decode, segments, velocities


def corners(state, length=4.5, width=1.8):
    theta = state[2]
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.array([[length/2, width/2], [length/2, -width/2], [-length/2, -width/2], [-length/2, width/2]]) @ rot.T + state[:2]


def overlap(a, b, length=4.5, width=1.8):
    ca, cb = corners(a, length, width), corners(b, length, width)
    for poly in (ca, cb):
        for edge in (poly[1] - poly[0], poly[2] - poly[1]):
            axis = np.array([-edge[1], edge[0]])
            pa, pb = ca @ axis, cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def pair_ttc(a, b, va, vb, max_time=10.):

    ca, cb = corners(a), corners(b)
    enter, leave = 0., max_time
    for poly in (ca, cb):
        for edge in (poly[1] - poly[0], poly[2] - poly[1]):
            axis = np.array([-edge[1], edge[0]])
            pa, pb = ca @ axis, cb @ axis
            speed = float((vb - va) @ axis)
            if abs(speed) < 1e-9:
                if pa.max() < pb.min() or pb.max() < pa.min():
                    return None
                continue
            t1, t2 = (pa.min() - pb.max()) / speed, (pa.max() - pb.min()) / speed
            enter, leave = max(enter, min(t1, t2)), min(leave, max(t1, t2))
            if enter > leave:
                return None
    return float(enter) if leave >= 0 and enter <= max_time else None


def minimum_ttc(states, valid, dt=.1):
    v = velocities(states, valid, dt)
    result = None
    for t in range(valid.shape[1]):
        ids = np.flatnonzero(valid[:, t])
        for k, i in enumerate(ids):
            for j in ids[k + 1:]:
                value = pair_ttc(states[i, t], states[j, t], v[i, t], v[j, t])
                if value is not None:
                    result = value if result is None else min(value, result)
    return result


def collision_flags(states, valid):
    flags = np.zeros_like(valid)
    for t in range(valid.shape[1]):
        ids = np.flatnonzero(valid[:, t])
        for ii, i in enumerate(ids):
            for j in ids[ii + 1:]:
                if np.linalg.norm(states[i, t, :2] - states[j, t, :2]) < 6 and overlap(states[i, t], states[j, t]):
                    flags[i, t] = flags[j, t] = True
    return flags


def offroad_flags(states, valid, road):
    result = np.zeros_like(valid)
    for i, t in zip(*np.nonzero(valid)):
        c = corners(states[i, t])


        b = road.bounds(c[:, 0])
        result[i, t] = ((c[:, 1] > b[0]) | (c[:, 1] < b[-1])).any()
    return result


def kinematics(states, valid, dt=.1, amax=10.4, kmax=.3):

    n = len(states)
    scorable, violation, failure = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        if not valid[i].any():
            continue
        if not np.isfinite(states[i][valid[i]]).all():
            failure[i] = True
            continue
        for a, b in segments(valid[i]):
            if b - a < 3:
                continue
            scorable[i] = True
            s = states[i, a:b]
            v = np.gradient(s[:, :2], dt, axis=0)
            speed = np.linalg.norm(v, axis=-1)
            accel = np.diff(speed) / dt
            heading_next = np.where(speed[1:] > .6, np.arctan2(v[1:, 1], v[1:, 0]), s[1:, 2])
            angle = (heading_next - s[:-1, 2] + np.pi) % (2 * np.pi) - np.pi
            distance = speed[:-1] * dt + .5 * accel * dt ** 2
            curvature = np.divide(angle, distance, out=np.zeros_like(angle), where=(np.minimum(speed[:-1], speed[1:]) >= .6))
            violation[i] |= ((abs(accel) > amax + .001) | (abs(curvature) > kmax + .001)).any()
        failure[i] |= not scorable[i]
    return dict(scorable=scorable, violation=violation, failure=failure)


def evaluate(tracks, road):
    tracks = np.asarray(tracks)
    total = fail = kv_n = kv_bad = no_segment = empty = malformed_n = coll_n = off_n = 0
    for x in tracks:
        states, valid, malformed = decode(x.transpose(1, 2, 0))
        active = valid.any(-1) | malformed.any(-1)
        collision = collision_flags(states, valid).any(-1)
        offroad = offroad_flags(states, valid, road).any(-1)
        kin = kinematics(states, valid)
        total += int(active.sum())
        fail += int((active & (collision | offroad | malformed.any(-1))).sum())
        nonfinite = ~np.isfinite(x).all(axis=(0, 2))
        scorable = kin['scorable'] & ~nonfinite
        kv_n += int(scorable.sum())
        kv_bad += int((kin['violation'] & scorable).sum())
        no_segment += int((active & ~scorable).sum())
        empty += int(not active.any())
        malformed_n += int(malformed.any(-1).sum())
        coll_n += int(collision.sum())
        off_n += int(offroad.sum())
    return dict(scenarios=len(tracks), trajectories=total, fail_pct=100 * fail / total if total else None,
                collision_trajectories=coll_n, offroad_trajectories=off_n, malformed_trajectories=malformed_n,
                kv_pct=100 * kv_bad / kv_n if kv_n else None, kv_scorable=kv_n,
                unscorable_nonempty_trajectories=no_segment, empty_scenarios=empty,
                note='Raw trajectories, 4.5m×1.8m boxes')


def frechet(real, generated):

    mr, mg = real.mean(0), generated.mean(0)
    cr = np.atleast_2d(np.cov(real, rowvar=False)) + np.eye(real.shape[1]) * 1e-6
    cg = np.atleast_2d(np.cov(generated, rowvar=False)) + np.eye(real.shape[1]) * 1e-6
    values, vectors = np.linalg.eigh(cr)
    root = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    eig = np.linalg.eigvalsh(root @ cg @ root)
    return float(max(0, ((mr - mg) ** 2).sum() + np.trace(cr + cg) - 2 * np.sqrt(np.maximum(eig, 0)).sum()))


def continuity_features(tracks, boundaries, dt=.1):
    accels, jerks = [], []
    for x in np.asarray(tracks):
        s, valid, _ = decode(x.transpose(1, 2, 0))
        for i in range(len(s)):
            for a, b in segments(valid[i]):
                if b - a < 5:
                    continue
                v = np.gradient(s[i, a:b, :2], dt, axis=0)
                acc = np.gradient(v, dt, axis=0)
                jerk = np.gradient(acc, dt, axis=0)
                for t in boundaries:
                    if a + 2 <= t < b - 2:
                        accels.append(acc[t - a]); jerks.append(jerk[t - a])
    return np.asarray(accels).reshape(-1, 2), np.asarray(jerks).reshape(-1, 2)


def continuity_score(real, generated, boundaries):
    r = continuity_features(real, boundaries)
    g = continuity_features(generated, boundaries)
    if any(len(x) < 2 for x in (*r, *g)):
        return dict(kcs=None, reason='Insufficient boundary observations')
    distances = [frechet(a, b) / max(float(np.var(a, axis=0, ddof=1).sum()), 1e-6) for a, b in zip(r, g)]
    return dict(kcs=float(np.exp(-.5 * (.7 * distances[0] + .3 * distances[1]))),
                normalized_acceleration_fd=distances[0], normalized_jerk_fd=distances[1],
                convention='Boundary acceleration/jerk, covariance-normalized FD')


def pool_episodes(episodes):
    n = len(episodes)
    nc = sum(bool(e['collision']) for e in episodes)
    unknown_fault = sum(e['collision'] and e.get('at_fault') is None for e in episodes)
    verified = [e.get('verified', e.get('verification', {}).get('verified')) for e in episodes]
    unknown_verification = sum(e['collision'] and v is None for e, v in zip(episodes, verified))
    na = sum(bool(e['collision'] and e.get('at_fault')) for e in episodes)
    nvc = sum(bool(e['collision'] and v) for e, v in zip(episodes, verified))
    nva = sum(bool(e['collision'] and e.get('at_fault') and v) for e, v in zip(episodes, verified))
    ratio = lambda a, b: 100 * a / b if b else None
    return dict(episodes=n, collisions=nc, at_fault=None if unknown_fault else na,
                verified_collisions=None if unknown_verification else nvc,
                verified_at_fault=None if unknown_fault or unknown_verification else nva,
                unknown_fault_episodes=int(unknown_fault), unknown_verification_episodes=int(unknown_verification),
                collision_pct=ratio(nc, n), acr_pct=None if unknown_fault else ratio(na, n),
                varc_pct=None if unknown_verification else ratio(nvc, nc),
                varaf_pct=None if unknown_fault or unknown_verification else ratio(nva, na),
                vafr_pct=None if unknown_fault or unknown_verification else ratio(nva, n),
                aggregation='Pooled counts; undefined ratios:null')
