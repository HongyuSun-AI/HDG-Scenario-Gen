import numpy as np
from .representation import decode

POLICY=dict(version='padding_exit_v1',padding='Padding<-0.9 retires actor',malformed='Keep actor and brake',unknown_speed='Peer-median speed; nearest/zero fallback')

def estimate_unknown_speeds(states,valid,speeds,known,road):
    result=np.array(speeds,copy=True,dtype=float);audit=[]
    donors=np.flatnonzero(valid & known)
    for j in np.flatnonzero(valid & ~known):
        lane=int(road.lane(states[j,0],states[j,1]))
        peers=[int(i) for i in donors if int(road.lane(states[i,0],states[i,1]))==lane and np.cos(states[i,2]-states[j,2])>.5 and np.linalg.norm(states[i,:2]-states[j,:2])<=40]
        method='same_lane_median'
        if not peers:
            compatible=[int(i) for i in donors if np.cos(states[i,2]-states[j,2])>.5]
            peers=[min(compatible,key=lambda i:np.linalg.norm(states[i,:2]-states[j,:2]))] if compatible else []
            method='nearest_known' if peers else 'zero_no_evidence'
        result[j]=float(np.median(np.asarray(speeds)[peers])) if peers else 0.
        audit.append(dict(vehicle=int(j),method=method,donors=peers,estimated_speed=float(result[j])))
    return result,audit


def birth_speed(plan,index,t,states,valid,speeds,road,dt=.1):


    if t+1<plan.shape[1]:
        pair,v,bad=decode(plan[index,t:t+2])
        if v.all():return float(min(35.,np.linalg.norm(pair[1,:2]-pair[0,:2])/dt))
    available=valid.copy();available[index]=True;known=valid.copy();known[index]=False
    return float(estimate_unknown_speeds(states,available,speeds,known,road)[0][index])
