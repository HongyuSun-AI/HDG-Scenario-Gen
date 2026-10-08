import numpy as np
from .representation import LOW,HIGH,segments
from .closed_loop import evaluation_end

def overlap_matrix(states):
    heading=states[:,2];c=np.cos(heading);s=np.sin(heading)
    axes=np.stack([np.stack([c,s],1),np.stack([-s,c],1)],1)
    delta=states[None,:,:2]-states[:,None,:2]
    distance=np.abs(np.einsum('ijc,ikc->ijk',delta,axes))
    dot=np.abs(np.einsum('ikc,jlc->ijkl',axes,axes))
    radii=np.array([2.25,.9])[None,None,:]+(dot*np.array([2.25,.9])[None,None,None,:]).sum(-1)
    side=(distance<=radii).all(-1)
    return side&side.T

def analyze(e):
    tr=e['tracks'];tr=tr.numpy() if hasattr(tr,'numpy') else tr
    end=evaluation_end(e,tr.shape[-1]);states=tr[:,:,:end].transpose(1,2,0)*(HIGH-LOW)+LOW
    valid=np.isfinite(states).all(-1)&(tr[:,:,:end]>=-.1).all(0);valid[e['ego']]=False
    collisions=[];pair_frames=0;birth_pairs=0;exposure=0;seen=set();jerks=[];yaw_rates=[];boundary_jerks=[]
    for t in range(end):
        ids=np.flatnonzero(valid[:,t]);exposure+=len(ids)
        if len(ids)<2:continue
        mat=overlap_matrix(states[ids,t]);ii,jj=np.where(np.triu(mat,1));pair_frames+=len(ii)
        for i,j in zip(ids[ii],ids[jj]):
            pair=(int(i),int(j))
            if pair not in seen:
                seen.add(pair);born=t>0 and (not valid[i,t-1] or not valid[j,t-1]);birth_pairs+=int(born)
                collisions.append(dict(frame=t,vehicles=list(pair),on_birth=bool(born)))
    for i in range(len(states)):
        for a,b in segments(valid[i]):
            if b-a<4:continue
            pos=states[i,a:b,:2];speed=np.linalg.norm(np.diff(pos,axis=0),axis=1)/.1
            jerk=np.diff(speed,n=2)/.01;jerks.extend(np.abs(jerk).tolist())
            yaw=np.diff(np.unwrap(states[i,a:b,2]))/.1;yaw_rates.extend(np.abs(yaw).tolist())
            for k,value in enumerate(jerk):
                if any(abs((a+k+2)-cut)<=1 for cut in e.get('boundaries',[])):boundary_jerks.append(abs(float(value)))
    return dict(sv_sv_collision=bool(seen),unique_sv_pairs=len(seen),sv_pair_frames=pair_frames,birth_collision_pairs=birth_pairs,sv_state_frames=exposure,events=collisions,
        abs_jerk=jerks,abs_yaw_rate=yaw_rates,boundary_abs_jerk=boundary_jerks)

def summarize(items):
    r={k:sum(x[k] for x in items) for k in ['sv_sv_collision','unique_sv_pairs','sv_pair_frames','birth_collision_pairs','sv_state_frames']};r['episodes']=len(items)
    for k in ['abs_jerk','abs_yaw_rate','boundary_abs_jerk']:
        a=np.array([v for x in items for v in x[k]]);r[k]=dict(mean=float(a.mean()),p95=float(np.percentile(a,95)),max=float(a.max())) if len(a) else None
    return r
