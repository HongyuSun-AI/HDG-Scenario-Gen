import numpy as np
import torch
from .representation import flatten_agents, unflatten_agents, history_mask, decode, normalize, LOW, HIGH
from .metrics import collision_flags, offroad_flags, kinematics
from .collision_attribution import CollisionRecorder


def outside_region(state, road):

    return bool(state[0] < road.xmin or state[0] > road.xmax)


def reached_destination(planner, state, destination):
    return getattr(planner, 'stop_at_destination', True) and np.linalg.norm(state[:2]-destination[:2]) < 2.


def evaluation_end(episode, horizon):

    limits = [horizon]
    for key in ['collision_time', 'arrival_time']:
        if episode.get(key) is not None: limits.append(episode[key]+1)
    if episode.get('exit_time') is not None: limits.append(episode['exit_time'])
    return min(limits)


def integrate(state, speed, acceleration, curvature, dt=.1):
    acceleration = np.clip(acceleration, -10.4, 10.4)
    next_speed = max(0., speed + acceleration * dt)
    distance = .5 * (speed + next_speed) * dt
    heading = state[2] + np.clip(curvature, -.3, .3) * distance
    next_state = np.array([state[0] + distance * np.cos(state[2]),
                           state[1] + distance * np.sin(state[2]),
                           (heading + np.pi) % (2 * np.pi) - np.pi])
    return next_state, next_speed


TRACKING_CONFIG = dict(version='bounded_reference_v2', speed_limit=35.,
    acceleration_limit=3., braking_limit=5., lateral_acceleration_limit=3.,
    position_gain=.5, maximum_speed_correction=2., response_time=1.)


def track_target(state, speed, target, dt=.1, previous_target=None):

    cfg = TRACKING_CONFIG
    if not np.isfinite(target).all():
        return integrate(state, speed, -5., 0., dt)
    offset = target[:2] - state[:2]
    forward = np.array([np.cos(target[2]), np.sin(target[2])])

    reference_speed = speed
    if previous_target is not None and np.isfinite(previous_target).all():
        reference_speed = max(0., float((target[:2]-previous_target[:2]) @ forward) / dt)
    reference_speed = min(reference_speed, cfg['speed_limit'])
    correction = np.clip(cfg['position_gain'] * (offset @ forward - reference_speed * dt),
                         -cfg['maximum_speed_correction'], cfg['maximum_speed_correction'])
    desired_speed = np.clip(reference_speed + correction, 0., cfg['speed_limit'])
    normal = np.array([-forward[1], forward[0]])
    desired_heading = target[2] + np.arctan2(offset @ normal, max(5., speed))
    error = (desired_heading - state[2] + np.pi) % (2 * np.pi) - np.pi
    acceleration = np.clip((desired_speed-speed)/cfg['response_time'],
                           -cfg['braking_limit'], cfg['acceleration_limit'])
    peak_speed = max(speed, speed+acceleration*dt, 0.)
    curvature_limit = min(.3, cfg['lateral_acceleration_limit']/max(peak_speed**2, 1.))
    curvature = np.clip(error/max(speed*cfg['response_time'], 1.),
                        -curvature_limit, curvature_limit)
    return integrate(state, speed, acceleration, curvature, dt)


class ReactivePlanner:
    name = 'reactive_demo'

    def __init__(self, road, cruise_speed=12.):
        self.road, self.cruise_speed = road, cruise_speed

    def reset(self, initial, ego, destination):
        self.ego = ego
        self.destination = np.array(destination)

    def step(self, observed, valid, speed, dt, future_env=None):
        s = observed[self.ego]
        lane = int(self.road.lane(s[0], s[1]))
        lane = max(0, lane)
        b = self.road.bounds(min(s[0] + 15, self.road.xmax))
        if lane == 2 and b[2] - b[3] < 2.:
            lane = 1
        target_speed = min(self.cruise_speed, np.sqrt(max(0., 2 * 3.4 * (self.destination[0] - s[0]))))
        ids = np.flatnonzero(valid & (np.arange(len(valid)) != self.ego))
        for j in ids:
            dx = observed[j, 0] - s[0]
            if dx > 0 and abs(observed[j, 1] - s[1]) < 2.2:
                target_speed = min(target_speed, max(0., (dx - 6.) / 1.5))

        if future_env is not None:
            for ahead, env in enumerate(future_env[:20], 1):
                projected = s[:2] + speed * ahead * dt * np.array([np.cos(s[2]), np.sin(s[2])])
                for j in ids:
                    if np.isfinite(env[j]).all() and abs(env[j, 1] - projected[1]) < 2.3 and abs(env[j, 0] - projected[0]) < 5.:
                        target_speed = 0.
        desired_heading = np.arctan2(self.road.center(lane, min(s[0] + 10, self.road.xmax)) - s[1], 10.)
        error = (desired_heading - s[2] + np.pi) % (2 * np.pi) - np.pi
        return integrate(s, speed, (target_speed - speed) / .5, error / max(speed, 1.), dt)


def run_closed_loop(model, diffusion, initial, first_states, initial_speeds, risk,
                    ego, planner, destination, horizon, execution=10, cfg=1., dt=.1, sv_controller='trajectory'):

    from .sv_front_guard import apply_front_guard, controller_metadata
    from .sv_lifecycle import POLICY, birth_speed
    controller_info=controller_metadata(sv_controller,TRACKING_CONFIG)
    sv_interventions=[]
    lifecycle_events=[]
    device = next(model.parameters()).device
    n = len(first_states)
    from .sv_stable import StableSVController
    stable=StableSVController(planner.road,n) if sv_controller=='congestion_stable' else None
    realized = np.full((n, horizon, 3), -1., dtype=np.float32)
    realized[:, 0] = first_states
    states, valid, malformed = decode(first_states)
    speeds = np.asarray(initial_speeds).copy()
    planner.reset(states.copy(), ego, destination)
    if hasattr(planner, 'set_initial_speeds'):
        planner.set_initial_speeds(speeds)
    recorder = CollisionRecorder(ego, planner.road)
    recorder.observe(states, valid, speeds, 0)
    collision_time = 0 if recorder.events else None
    arrival_time = 0 if reached_destination(planner, states[ego], destination) else None
    exit_time = None
    exited = np.zeros(n, dtype=bool)
    p, boundaries, history_errors = 1, [], []
    last_plan = None
    invalid_targets = 0
    while p < horizon:
        stopped = collision_time is not None or arrival_time is not None or exit_time is not None
        if not stopped or last_plan is None:
            tensor = torch.from_numpy(realized.transpose(2, 0, 1)[None]).to(device)
            history = flatten_agents(tensor)
            mask = history_mask(torch.tensor([p], device=device), horizon)
            plan = diffusion.sample(model, tuple(history.shape), torch.tensor([risk], device=device),
                                    history, mask, initial.to(device), cfg=cfg)
            history_errors.append(float((plan[..., :p] - history[..., :p]).abs().max()))
            last_plan = unflatten_agents(plan).cpu().numpy()[0].transpose(1, 2, 0)
            boundaries.append(p)
        end = min(horizon, p + execution) if not stopped else horizon
        for t in range(p, end):
            targets, target_valid, bad = decode(last_plan[:, t])
            invalid_targets += int(bad.sum())
            observed_speeds=speeds.copy()
            current = states.copy()
            active = valid.copy()
            new_states, new_valid = states.copy(), valid.copy()
            if stable is not None:
                proposals,proposal_speeds,events=stable.propose(current,active,observed_speeds,last_plan,t,ego,dt)
                sv_interventions.extend(dict(frame=t,**event) for event in events)
            for j in range(n):
                if j == ego:
                    continue
                if valid[j] and (last_plan[j,t]<-.9).all():
                    new_valid[j]=False; exited[j]=True
                    lifecycle_events.append(dict(frame=t,vehicle=j,reason='generated_padding'))
                    continue
                if valid[j]:
                    previous_target = decode(last_plan[j, t-1])[0]
                    if stable is not None:
                        new_states[j],speeds[j]=proposals[j],proposal_speeds[j]
                    else:
                        new_states[j], speeds[j] = track_target(states[j], speeds[j], targets[j], dt, previous_target)
                    if sv_controller=='front_guard':
                        new_states[j],speeds[j],audit=apply_front_guard(j,current,active,observed_speeds,new_states[j],speeds[j],dt)
                        if audit is not None:sv_interventions.append(dict(frame=t,**audit))
                    if outside_region(new_states[j], planner.road):
                        new_valid[j] = False; exited[j] = True
                elif target_valid[j] and not exited[j] and not outside_region(targets[j], planner.road):
                    new_states[j], new_valid[j] = targets[j], True
                    spawn_states=current.copy();spawn_states[j]=targets[j]
                    speeds[j]=birth_speed(last_plan,j,t,spawn_states,active,observed_speeds,planner.road,dt)
            if collision_time is None and arrival_time is None and exit_time is None:
                new_states[ego], speeds[ego] = planner.step(current, active, speeds[ego], dt)
            if exit_time is None and outside_region(new_states[ego], planner.road):
                exit_time = t; new_valid[ego] = False
            states, valid = new_states, new_valid
            realized[:, t] = normalize(states, valid)


            if collision_time is None and arrival_time is None and exit_time is None:
                recorder.observe(states, valid, speeds, t)
                if recorder.events:
                    collision_time = t
            if collision_time is None and arrival_time is None and exit_time is None and reached_destination(planner, states[ego], destination):
                arrival_time = t
        p = end
    attribution = recorder.result()
    return dict(tracks=realized.transpose(2, 0, 1), boundaries=boundaries,
                history_max_error=max(history_errors, default=0.), ego=int(ego),
                collision=collision_time is not None, collision_time=collision_time,
                arrival_time=arrival_time, exit_time=exit_time, invalid_generated_targets=invalid_targets,
                lifecycle_policy=POLICY,lifecycle_events=lifecycle_events,
                surrounding_controller=controller_info,sv_interventions=sv_interventions,
                planner=planner.name, at_fault=attribution['at_fault'], attribution=attribution,
                planner_config=planner.describe() if hasattr(planner, 'describe') else {'name':planner.name},
                planner_audits=list(getattr(planner, 'audits', [])))


def verify_fixed_replay(episode, initial_speeds, planner, destination, dt=.1):

    raw = episode['tracks'].transpose(1, 2, 0)
    _, valid, malformed = decode(raw)
    replay = raw.astype(np.float64)*(HIGH-LOW)+LOW
    replay[~valid] = np.nan
    ego = episode['ego']
    candidate = replay.copy()
    planner_config = planner.describe() if hasattr(planner, 'describe') else {'name':planner.name}
    if episode.get('planner_config', planner_config) != planner_config:
        raise ValueError('Verification planner mismatch')
    planner.reset(candidate[:, 0].copy(), ego, destination)
    if hasattr(planner, 'set_initial_speeds'):
        planner.set_initial_speeds(initial_speeds)
    speed = float(initial_speeds[ego])
    limit = min(raw.shape[1], episode.get('exit_time') if episode.get('exit_time') is not None else raw.shape[1])
    end = limit
    verification_exit = None
    for t in range(1, limit):
        candidate[ego, t], speed = planner.step(candidate[:, t - 1].copy(), valid[:, t - 1], speed, dt,
                                                future_env=replay[:, t:limit].transpose(1, 0, 2))
        if outside_region(candidate[ego,t], planner.road):
            end = t; verification_exit = t; break
    scored, scored_valid = candidate[:,:end], valid[:,:end]
    collision = bool(collision_flags(scored, scored_valid)[ego].any())
    offroad = bool(offroad_flags(scored, scored_valid, planner.road)[ego].any())
    kv = kinematics(scored[ego:ego+1], scored_valid[ego:ego+1], dt)
    complete = bool(end > 0 and scored_valid[ego].all() and not malformed[:,:end].any())
    success = complete and not collision and not offroad and bool(kv['scorable'][0]) and not bool(kv['violation'][0])
    others = np.arange(len(replay)) != ego
    return dict(verified=success, collision=collision, offroad=offroad,
                kinematic_violation=bool(kv['violation'][0]), complete=complete,
                scope=f'Fixed-replay witness: {planner.name}',
                evaluated_frames=end, exit_time=verification_exit,
                planner_config=planner_config, planner_audits=list(getattr(planner, 'audits', [])),
                ego_trajectory=candidate[ego,:end].tolist())
