import json
import time
from pathlib import Path
import numpy as np
import torch
from .full_training import legacy_load, torch_write, json_write
from .relabel import sha256
from .representation import decode, velocities, normalize
from .risk import assess_risk_vectorized
from .road import MergeRoad


def prepare_congestion(source, output, map_path, kappa=1.6, labels_path=None):

    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    source_hash=sha256(source)
    data_path=output/'data.pt'
    if data_path.exists():
        data=torch.load(data_path,weights_only=True,map_location='cpu')
        if (data['metadata']['source_sha256'] != source_hash or data['metadata']['kappa'] != kappa
                or data['metadata'].get('map_sha256') != sha256(map_path)):
            raise ValueError('Prepared data fingerprint differs')
        return data
    old=legacy_load(source); tracks=old.data.float()
    n,c,agents,horizon=tracks.shape
    labels=None
    if labels_path:
        labels=torch.load(labels_path,weights_only=True,map_location='cpu')
        if (labels['metadata']['source_sha256']!=source_hash or
            labels['metadata']['parameters']['kappa']!=kappa or
            labels['metadata']['map_sha256']!=sha256(map_path) or
            not torch.equal(labels['source_indices'],torch.arange(n))):
            raise ValueError('Calibrated labels mismatch')
    road=MergeRoad(map_path)
    continuous=np.empty((n,agents),np.float32); initial=np.empty((n,3,28,3),np.float32)
    valid=(tracks >= 0).all(1)
    order=torch.argsort(valid.sum(-1),dim=1,descending=True,stable=True)
    tracks=tracks.gather(2,order[:,None,:,None].expand(-1,3,-1,horizon))
    start=time.perf_counter()
    for i in range(n):
        states,present,bad=decode(tracks[i].permute(1,2,0).numpy())
        continuous[i]=(labels['agent_risk'][i,order[i]].numpy() if labels is not None else
                       assess_risk_vectorized(states,present,road,kappa=kappa))
        initial[i]=road.layout(states[:,0],present[:,0])[0]
        if (i+1)%500 == 0: print(f'prepare {Path(source).stem} {i+1}/{n}',flush=True)
    scores=torch.from_numpy(continuous.max(1)); classes=(scores >= .9).float()
    metadata=dict(source=str(source),source_sha256=source_hash,map_path=str(map_path),
        map_sha256=sha256(map_path),scenarios=n,agents=agents,horizon=horizon,dt=.1,kappa=kappa,
        risk_parameters={'kappa':kappa},label_rule='risk>=0.9:1; risk<0.9:0',
        risk_labels_sidecar=str(labels_path) if labels_path else None,
        coverage='Full source, 140 frames',
        positive_count=int(classes.sum()),negative_count=int((classes==0).sum()),
        equal_count=int((scores==.9).sum()),seconds=time.perf_counter()-start)
    data=dict(tracks=tracks,initial=torch.from_numpy(initial),risk=classes,continuous_risk=scores,
              agent_risk=torch.from_numpy(continuous),source_indices=torch.arange(n),order=order,metadata=metadata)
    torch_write(data_path,data); json_write(output/'data_manifest.json',metadata)
    return data


def select_initializations(data, groups=10, seed=20261007, ego_x_fraction_max=None):

    rng=np.random.default_rng(seed); road=MergeRoad(data['metadata']['map_path'])
    scores=data['continuous_risk'].numpy()
    eligible=np.flatnonzero(scores >= .9); rng.shuffle(eligible)
    selected=[]; rejected=[]
    for idx in eligible:
        states,valid,bad=decode(data['tracks'][idx].permute(1,2,0).numpy())
        speeds=np.linalg.norm(velocities(states,valid),axis=-1)
        agent_risk=data['agent_risk'][idx].numpy()
        choices=[]
        for frame in range(states.shape[1]-2):
            ids=np.flatnonzero(valid[:,frame] & (agent_risk >= .9))
            if ego_x_fraction_max is not None:
                ids=ids[(states[ids,frame,0]>=road.xmin) & (states[ids,frame,0]<=road.xmin+(road.xmax-road.xmin)*ego_x_fraction_max)]
            if not len(ids): continue
            ego=int(ids[np.argmax(agent_risk[ids])])
            future=np.flatnonzero(valid[ego,frame:])+frame
            if len(future)<3: continue
            goal=states[ego,future[-1]]
            if goal[0]-states[ego,frame,0] <= 5.: continue
            if road.lane(states[ego,frame,0],states[ego,frame,1]) < 0: continue
            choices.append((frame,ego,int(future[-1])))
        if not choices:
            rejected.append(int(idx)); continue
        frame,ego,end=choices[int(rng.integers(len(choices)))]
        selected.append(dict(group=len(selected),dataset_index=int(idx),source_index=int(data['source_indices'][idx]),
            ego_original_row=int(data['order'][idx,ego]) if 'order' in data else ego,
            frame=frame,ego=ego,destination=states[ego,end].tolist(),initial_speeds=speeds[:,frame].tolist(),
            continuous_source_risk=float(scores[idx]),ego_source_risk=float(agent_risk[ego]),
            seed=int(seed+1000*(len(selected)+1))))
        if len(selected)==groups: break
    if len(selected)!=groups:
        raise ValueError(f'Only {len(selected)} eligible; requested {groups}')
    return dict(seed=seed,groups=selected,ego_x_fraction_max=ego_x_fraction_max,eligible_source_windows=len(eligible),rejected_no_valid_initial_frame=rejected,
        selection='Unique windows; random eligible frame',
        threshold_scope='Full-window risk>=0.9',
        independence='Distinct indices; recordings may overlap')


def initialization(data, item):
    idx,frame=item['dataset_index'],item['frame']
    first=data['tracks'][idx,:,:,frame].T.numpy().copy()
    states,valid,bad=decode(first)
    road=MergeRoad(data['metadata']['map_path'])
    initial=road.layout(states,valid)[0]
    reference=np.full_like(data['tracks'][idx].numpy(),-1.)
    reference[:,:,:140-frame]=data['tracks'][idx,:,:,frame:].numpy()
    from .sv_lifecycle import estimate_unknown_speeds
    all_valid=(data['tracks'][idx].numpy()>=-.1).all(0)
    known=all_valid[:,frame] & ((all_valid[:,frame-1] if frame>0 else False) | (all_valid[:,frame+1] if frame+1<140 else False))
    speeds,audit=estimate_unknown_speeds(states,valid,np.array(item['initial_speeds']),known,road)

    if audit:
        item.setdefault('original_initial_speeds',list(item['initial_speeds']))
        item['initial_speeds']=speeds.tolist();item['initial_speed_estimates']=audit
    return initial,first,speeds,item['ego'],np.array(item['destination']),reference


def check_high_risk_eligibility(source,map_path,output,groups=10,kappa=1.6):

    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    fingerprint=dict(source_sha256=sha256(source),map_sha256=sha256(map_path),kappa=kappa,threshold=.9,
        required_groups=groups,selection_version=1)
    if path.exists():
        saved=json.loads(path.read_text())
        if saved.get('fingerprint')==fingerprint: return saved
    old=legacy_load(source);road=MergeRoad(map_path);begin=time.perf_counter()
    high=[];usable=[];maximum=0.;max_index=None
    for i in range(len(old)):
        s,v,bad=decode(old.data[i].float().permute(1,2,0).numpy())
        risks=assess_risk_vectorized(s,v,road,kappa=kappa);score=float(risks.max())
        if score>maximum: maximum=score;max_index=i
        if score>=.9:
            high.append(i)
            for frame in range(138):
                ids=np.flatnonzero(v[:,frame] & (risks>=.9))
                if not len(ids): continue
                ego=int(ids[np.argmax(risks[ids])]);future=np.flatnonzero(v[ego,frame:])+frame
                if len(future)>=3 and s[ego,future[-1],0]-s[ego,frame,0]>5 and road.lane(s[ego,frame,0],s[ego,frame,1])>=0:
                    usable.append(i);break
        if (i+1)%1000==0: print(f'Eligibility {i+1}/{len(old)}: high={len(high)}, usable={len(usable)}',flush=True)
        if len(usable)>=groups: break
    report=dict(fingerprint=fingerprint,kappa=kappa,threshold=.9,checked=i+1,total=len(old),full_scan=i+1==len(old),
        high_risk_count_found=len(high),usable_count_found=len(usable),high_risk_indices=high,usable_indices=usable,
        maximum_risk_seen=maximum,maximum_source_index=max_index,seconds=time.perf_counter()-begin,
        labels_written=False,source_modified=False,eligible_for_10_groups=len(usable)>=groups)
    json_write(path,report)
    return report
