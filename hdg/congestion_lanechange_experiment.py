import numpy as np
from .representation import decode,velocities
from .metrics import overlap
from .left_lanechange_experiment import occupied_lane

def candidates(data,road,seed):
 result=[]
 eligible=np.flatnonzero(data['continuous_risk'].numpy()>=.9);np.random.default_rng(seed).shuffle(eligible)
 for idx in eligible:
  states,valid,bad=decode(data['tracks'][idx].permute(1,2,0).numpy());options=[]
  for ego in np.flatnonzero(valid.any(axis=1)):
   lanes=[occupied_lane(s,road) if valid[ego,t] else -1 for t,s in enumerate(states[ego])]
   for end in range(4,len(lanes)):
    if not all(l==1 for l in lanes[end-4:end+1]):continue
    starts=[t for t in range(end-4) if lanes[t]==2 and valid[ego,t:t+3].all()]
    for frame in reversed(starts):
     if any(overlap(states[ego,frame],states[j,frame]) for j in np.flatnonzero(valid[:,frame]) if j!=ego):continue
     options.append((end-frame,-float(data['agent_risk'][idx,ego]),frame,int(ego),end));break
    if starts:break
  if not options:continue
  _,_,frame,ego,end=min(options);speeds=np.linalg.norm(velocities(states,valid),axis=-1)
  future=np.flatnonzero(valid[ego,frame:])+frame
  result.append(dict(group=len(result),dataset_index=int(idx),source_index=int(data['source_indices'][idx]),ego_original_row=int(data['order'][idx,ego]),frame=frame,ego=ego,
    destination=states[ego,future[-1]].tolist(),initial_speeds=speeds[:,frame].tolist(),continuous_source_risk=float(data['continuous_risk'][idx]),ego_source_risk=float(data['agent_risk'][idx,ego]),
    initial_x=float(states[ego,frame,0]),initial_lane=2,reference_lane_change_confirmed_frame=int(end),seed=int(seed+1000*(len(result)+1))))
 return result
