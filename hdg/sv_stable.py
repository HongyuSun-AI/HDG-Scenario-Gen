import numpy as np
from .representation import decode
from .sv_front_guard import apply_front_guard
from .sv_diagnostics import overlap_matrix

CONFIG=dict(version='congestion_preview_v2',preview_frames=6,jerk_limit=3.,
            emergency_jerk_limit=20.,curvature_rate_limit=.12,yaw_rate_limit=.35,
            lateral_acceleration_limit=2.5,prediction_seconds=1.2,
            sv_conflict_braking=4.,reference_priority=True,
            ego_interaction='Front guard; SV-SV yielding')

def wrap(a):return (a+np.pi)%(2*np.pi)-np.pi

class StableSVController:
 def __init__(self,road,n):
  self.road=road;self.acceleration=np.zeros(n);self.curvature=np.zeros(n)
 def road_heading(self,x):
  return np.arctan((self.road.center(1,x+.5)-self.road.center(1,x-.5)))
 def propose(self,states,valid,speeds,plan,t,ego,dt=.1):
  from .closed_loop import integrate
  ids=np.flatnonzero(valid & ~(plan[:,t]<-.9).all(-1));sv=ids[ids!=ego]
  acc=np.zeros(len(states));curv=np.zeros(len(states));audit=[]
  for j in sv:
   s=states[j];speed=speeds[j];rh=float(self.road_heading(s[0]));forward=np.array([np.cos(rh),np.sin(rh)])
   future,present,_=decode(plan[j,t:min(t+CONFIG['preview_frames'],plan.shape[1])])

   missing=np.flatnonzero(~present);count=int(missing[0]) if len(missing) else len(present)
   if not count:
    desired_acc=-4.;desired_k=0.
   else:
    f=future[:count];target=f[-1];previous=decode(plan[j,max(0,t-1)])[0]
    points=np.vstack([previous,f]) if np.isfinite(previous).all() else f
    displacement=np.diff(points[:,:2],axis=0)
    along=displacement@forward
    ref_speed=float(np.clip(np.median(along)/dt,0,20)) if len(along) else speed
    desired_speed=np.clip(ref_speed+np.clip(.25*((f[0,:2]-s[:2])@forward-ref_speed*dt),-1,1),0,20)
    desired_acc=np.clip((desired_speed-speed)/1.2,-5,3)

    route=target[:2]-s[:2];long=max(float(route@forward),max(3.,speed*.8))
    normal=np.array([-forward[1],forward[0]])
    desired_heading=rh+np.arctan2(float(route@normal),long)
    desired_k=wrap(desired_heading-s[2])/max(speed*1.2,2.)
   acc[j]=np.clip(desired_acc,self.acceleration[j]-3*dt,self.acceleration[j]+3*dt)
   limit=min(.2,CONFIG['yaw_rate_limit']/max(speed,.5),2.5/max(speed**2,1.))
   curv[j]=np.clip(np.clip(desired_k,self.curvature[j]-.12*dt,self.curvature[j]+.12*dt),-limit,limit)
   proposal,new_speed=integrate(s,speed,acc[j],curv[j],dt)
   _,guard_speed,event=apply_front_guard(j,states,valid,speeds,proposal,new_speed,dt)
   if event is not None:
    acc[j]=max(self.acceleration[j]-20*dt,(guard_speed-speed)/dt)
    audit.append(dict(kind='front_guard',**event))

  predicted=states.copy();velocity=speeds.copy();conflicts=set()
  for step in range(12):
   new_velocity=np.maximum(0,velocity+acc*dt);distance=.5*(velocity+new_velocity)*dt
   predicted[ids,:2]+=distance[ids,None]*np.stack([np.cos(predicted[ids,2]),np.sin(predicted[ids,2])],1)
   predicted[ids,2]=wrap(predicted[ids,2]+curv[ids]*distance[ids]);velocity=new_velocity
   if step not in (3,7,11):continue
   if len(sv)>1:
    pairs=np.argwhere(np.triu(overlap_matrix(predicted[sv]),1))
    conflicts.update((int(sv[a]),int(sv[b])) for a,b in pairs)
  yields={}
  for a,b in sorted(conflicts):
   rh=float(self.road_heading(.5*(states[a,0]+states[b,0])))
   forward=np.array([np.cos(rh),np.sin(rh)]);long=float((states[b,:2]-states[a,:2])@forward)
   if abs(long)>1.:
    loser,winner=(a,b) if long>0 else (b,a)
   else:

    turn_a=abs(wrap(states[a,2]-rh)+curv[a]*max(speeds[a],.5))
    turn_b=abs(wrap(states[b,2]-rh)+curv[b]*max(speeds[b],.5))
    loser,winner=(a,b) if (turn_a,a)>(turn_b,b) else (b,a)
   yields[loser]=winner
  for j,winner in yields.items():
   acc[j]=min(acc[j],max(-4.,self.acceleration[j]-20*dt))

   desired_k=wrap(float(self.road_heading(states[j,0]))-states[j,2])/max(speeds[j]*.8,2.)
   limit=min(.2,.35/max(speeds[j],.5),2.5/max(speeds[j]**2,1.))
   curv[j]=np.clip(np.clip(desired_k,self.curvature[j]-.12*dt,self.curvature[j]+.12*dt),-limit,limit)
   audit.append(dict(kind='sv_predictive_yield',vehicle=int(j),leader=int(winner)))
  result=states.copy();result_speed=speeds.copy()
  for j in sv:
   result[j],result_speed[j]=integrate(states[j],speeds[j],acc[j],curv[j],dt)
   self.acceleration[j]=(result_speed[j]-speeds[j])/dt;self.curvature[j]=curv[j]
  return result,result_speed,audit
