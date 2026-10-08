import time
import numpy as np
import torch
from .closed_loop import track_target, TRACKING_CONFIG, outside_region, reached_destination
from .collision_attribution import CollisionRecorder
from .sv_front_guard import apply_front_guard, controller_metadata
from .sv_lifecycle import POLICY, birth_speed
from .representation import decode, normalize, flatten_agents, unflatten_agents, history_mask


def run_batch(model, diffusion, initial, first_states, initial_speeds, ego, destination,
              planner_factory, batch_size=32, horizon=140, execution=10, seed=7,
              risk=1., cfg=1., max_frames=None, on_progress=None, sv_controller='trajectory'):

    if horizon != model.horizon or execution < 1 or batch_size < 1:
        raise ValueError('Invalid batch/horizon/execution configuration')
    controller_info=controller_metadata(sv_controller,TRACKING_CONFIG)
    stop = horizon if max_frames is None else min(horizon, max_frames)
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    n = len(first_states)
    states, valid, bad = decode(first_states)
    contexts = []
    from .sv_stable import StableSVController
    for _ in range(batch_size):
        planner = planner_factory()
        planner.reset(states.copy(), ego, destination)
        if hasattr(planner, 'set_initial_speeds'):
            planner.set_initial_speeds(initial_speeds)
        recorder = CollisionRecorder(ego, planner.road)
        recorder.observe(states, valid, initial_speeds, 0)
        track = np.full((n, horizon, 3), -1., np.float32)
        track[:, 0] = first_states
        contexts.append(dict(planner=planner, recorder=recorder, states=states.copy(),
            sv_stable=StableSVController(planner.road,n) if sv_controller=='congestion_stable' else None,
            valid=valid.copy(), speeds=np.array(initial_speeds, copy=True), tracks=track,
            collision_time=0 if recorder.events else None,
            arrival_time=0 if reached_destination(planner, states[ego], destination) else None,
            exit_time=None, exited=np.zeros(n,dtype=bool),
            plan=None, boundaries=[], generation_windows=[], history_error=0., invalid_targets=0, sv_interventions=[],lifecycle_events=[]))
    conditions = torch.as_tensor(initial, dtype=torch.float32, device=device).reshape(1,3,28,3)
    sample_seconds = execution_seconds = 0.
    sample_batches = []
    for p in range(1, stop, execution):
        ids = [i for i,c in enumerate(contexts) if c['plan'] is None or
               (c['collision_time'] is None and c['arrival_time'] is None and c['exit_time'] is None)]
        if ids:
            history = flatten_agents(torch.from_numpy(np.stack(
                [contexts[i]['tracks'].transpose(2,0,1) for i in ids])).to(device))
            mask = history_mask(torch.full((len(ids),),p,device=device),horizon)
            if device.type == 'cuda': torch.cuda.synchronize()
            begin = time.perf_counter()
            plan = diffusion.sample(model,tuple(history.shape),torch.full((len(ids),),risk,device=device),
                history,mask,conditions.expand(len(ids),-1,-1,-1),cfg=cfg,generator=generator)
            if device.type == 'cuda': torch.cuda.synchronize()
            sample_seconds += time.perf_counter()-begin
            errors = (plan[...,:p]-history[...,:p]).abs().flatten(1).max(1).values.cpu().numpy()
            plans = unflatten_agents(plan).cpu().numpy()
            for j,i in enumerate(ids):
                c=contexts[i]; c['plan']=plans[j].transpose(1,2,0)
                c['history_error']=max(c['history_error'],float(errors[j]))
                c['boundaries'].append(p)
                c['generation_windows'].append(plans[j].copy())
            sample_batches.append(dict(frame=p,size=len(ids)))
        begin = time.perf_counter()
        for t in range(p,min(p+execution,stop)):
            for c in contexts:
                targets,target_valid,bad = decode(c['plan'][:,t])
                c['invalid_targets']+=int(bad.sum())
                previous,active = c['states'],c['valid']
                observed_speeds=c['speeds'].copy()
                current,present = previous.copy(),active.copy()
                if c['sv_stable'] is not None:
                    proposals,proposal_speeds,events=c['sv_stable'].propose(previous,active,observed_speeds,c['plan'],t,ego)
                    c['sv_interventions'].extend(dict(frame=t,**event) for event in events)
                for j in range(n):
                    if j == ego: continue
                    if active[j] and (c['plan'][j,t]<-.9).all():
                        present[j]=False; c['exited'][j]=True
                        c['lifecycle_events'].append(dict(frame=t,vehicle=j,reason='generated_padding'))
                        continue
                    if active[j]:
                        previous_target=decode(c['plan'][j,t-1])[0]
                        if c['sv_stable'] is not None:
                            current[j],c['speeds'][j]=proposals[j],proposal_speeds[j]
                        else:
                            current[j],c['speeds'][j]=track_target(previous[j],c['speeds'][j],targets[j],.1,previous_target)
                        if sv_controller=='front_guard':
                            current[j],c['speeds'][j],audit=apply_front_guard(j,previous,active,observed_speeds,current[j],c['speeds'][j],.1)
                            if audit is not None:c['sv_interventions'].append(dict(frame=t,**audit))
                        if outside_region(current[j],c['planner'].road):
                            present[j]=False; c['exited'][j]=True
                    elif target_valid[j] and not c['exited'][j] and not outside_region(targets[j],c['planner'].road):
                        current[j],present[j]=targets[j],True
                        spawn_states=previous.copy();spawn_states[j]=targets[j]
                        c['speeds'][j]=birth_speed(c['plan'],j,t,spawn_states,active,observed_speeds,c['planner'].road)
                running = c['collision_time'] is None and c['arrival_time'] is None and c['exit_time'] is None
                if running:
                    current[ego],c['speeds'][ego]=c['planner'].step(previous,active,c['speeds'][ego],.1)
                if running and outside_region(current[ego],c['planner'].road):
                    c['exit_time']=t; present[ego]=False; running=False
                c['states'],c['valid']=current,present
                c['tracks'][:,t]=normalize(current,present)
                if running:
                    c['recorder'].observe(current,present,c['speeds'],t)
                    if c['recorder'].events: c['collision_time']=t
                    if c['collision_time'] is None and reached_destination(c['planner'],current[ego],destination): c['arrival_time']=t
        execution_seconds+=time.perf_counter()-begin
        if on_progress: on_progress(dict(frame=min(p+execution,stop),sample_batches=sample_batches,
            sampling_seconds=sample_seconds,execution_seconds=execution_seconds))
    results=[]
    for c in contexts:
        attribution=c['recorder'].result()
        results.append(dict(tracks=c['tracks'].transpose(2,0,1),ego=int(ego),
            collision=c['collision_time'] is not None,collision_time=c['collision_time'],
            arrival_time=c['arrival_time'],exit_time=c['exit_time'],at_fault=attribution['at_fault'],attribution=attribution,
            boundaries=c['boundaries'],generation_windows=np.asarray(c['generation_windows']),
            history_max_error=c['history_error'],invalid_generated_targets=c['invalid_targets'],
            lifecycle_policy=POLICY,lifecycle_events=c['lifecycle_events'],
            surrounding_controller=controller_info,sv_interventions=c['sv_interventions'],
            planner=c['planner'].name,planner_config=c['planner'].describe() if hasattr(c['planner'],'describe') else {'name':c['planner'].name},
            planner_audits=list(getattr(c['planner'],'audits',[])),executed_frames=stop,
            preflight_only=stop != horizon))
    return results,dict(sampling_seconds=sample_seconds,execution_seconds=execution_seconds,
                       sample_batches=sample_batches,batch_size=batch_size,executed_frames=stop)
