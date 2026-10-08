import numpy as np
from .metrics import corners

CONFIG=dict(version='congestion_front_guard_v1',lookahead_m=20.,minimum_lateral_overlap_m=.9,
            minimum_heading_cosine=.5,standstill_gap_m=.75,short_headway_s=.35,
            ttc_trigger_s=1.2,closing_threshold_mps=.1,gap_response_s=.8,
            speed_response_s=.4,maximum_braking_mps2=4.)


def controller_metadata(mode,base):
    if mode=='congestion_stable':
        from .sv_stable import CONFIG as STABLE_CONFIG
        return dict(STABLE_CONFIG,base_controller=dict(base),front_guard=dict(CONFIG))
    return dict(base) if mode=='trajectory' else dict(CONFIG,base_controller=dict(base))


def apply_front_guard(index,states,valid,speeds,proposed_state,proposed_speed,dt=.1):

    from .closed_loop import integrate
    state=states[index];speed=float(speeds[index]);cfg=CONFIG
    forward=np.array([np.cos(state[2]),np.sin(state[2])]);normal=np.array([-forward[1],forward[0]])
    own=corners(state)-state[:2];own_lon=own@forward;own_lat=own@normal
    candidates=[]
    for j in np.flatnonzero(valid):
        if j==index or not np.isfinite(states[j]).all():continue
        delta=states[j,:2]-state[:2];long=float(delta@forward)
        if long<=0 or long>cfg['lookahead_m']:continue
        alignment=float(np.cos(states[j,2]-state[2]))
        if alignment<cfg['minimum_heading_cosine']:continue
        box=corners(states[j])-state[:2];lat=box@normal
        lateral_overlap=min(own_lat.max(),lat.max())-max(own_lat.min(),lat.min())
        if lateral_overlap<cfg['minimum_lateral_overlap_m']:continue
        gap=float((box@forward).min()-own_lon.max())
        leader_speed=max(0.,float(speeds[j])*alignment)
        candidates.append((gap,int(j),leader_speed))
    if not candidates:return proposed_state,proposed_speed,None
    gap,leader,leader_speed=min(candidates)
    closing=speed-leader_speed
    ttc=max(0.,gap)/closing if closing>cfg['closing_threshold_mps'] else None
    close=gap<cfg['standstill_gap_m']+cfg['short_headway_s']*speed
    approaching=ttc is not None and ttc<cfg['ttc_trigger_s']
    if not (close or approaching):return proposed_state,proposed_speed,None
    cap=leader_speed+max(0.,gap-cfg['standstill_gap_m'])/cfg['gap_response_s']
    base_acc=(proposed_speed-speed)/dt
    acceleration=max(-cfg['maximum_braking_mps2'],(cap-speed)/cfg['speed_response_s'])
    acceleration=min(base_acc,acceleration)
    if acceleration>=base_acc-1e-10:return proposed_state,proposed_speed,None
    distance=.5*(speed+proposed_speed)*dt
    angle=(proposed_state[2]-state[2]+np.pi)%(2*np.pi)-np.pi
    curvature=angle/distance if distance>1e-8 else 0.
    result,new_speed=integrate(state,speed,acceleration,curvature,dt)
    audit=dict(vehicle=int(index),leader=leader,gap_m=gap,ttc_s=ttc,
               original_acceleration=float(base_acc),applied_acceleration=float(acceleration),speed_cap=float(cap))
    return result,new_speed,audit
