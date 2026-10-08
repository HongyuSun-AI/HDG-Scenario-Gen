import hashlib
import numpy as np
import torch
from scipy.optimize import lsq_linear
from .generation_smoothing import smooth_future
from .road_projection import project_future_to_road, CONFIG as ROAD_PROJECTION_CONFIG
from .representation import LOW,HIGH,segments,flatten_agents,unflatten_agents

CONFIG=dict(dt=.1,position_scales_m=[3.,.75],acceleration_scale=3.,jerk_scale=3.,maximum_coordinate_correction_m=[3.,1.],road_half_width_margin=.9,history_frames=3,optimization='First contiguous future segment',heading='Displacement heading above 0.6m/s',constraints='Position and road-center bounds')

def optimize_future(tracks,cut,road):
 out=np.array(tracks,copy=True);valid=np.isfinite(tracks).all(0)&(tracks>=-.1).all(0)
 audit=dict(optimized=0,skipped=0,infeasible=0,solver_failed=0,max_correction_m=0.,correction_sum_m=0.,points=0)
 for agent in range(tracks.shape[1]):
  if cut<3 or not valid[agent,cut-3:cut].all():audit['skipped']+=1;continue
  n=0
  while cut+n<tracks.shape[2] and valid[agent,cut+n]:n+=1
  if n<4:audit['skipped']+=1;continue
  physical=tracks[:2,agent].T*(HIGH-LOW)[:2]+LOW[:2];history=physical[cut-3:cut];target=physical[cut:cut+n];origin=history[-1];h=history-origin;q=target-origin

  D2=np.diff(np.eye(n+3),n=2,axis=0)[1:]/.1**2
  D3=np.diff(np.eye(n+3),n=3,axis=0)/.1**3
  candidates=[];failed=False
  for dim in [0,1]:
   scale=CONFIG['position_scales_m'][dim]
   A=np.vstack([np.eye(n)/scale,D2[:,3:]/3.,D3[:,3:]/3.])
   rhs=np.r_[q[:,dim]/scale,-D2[:,:3]@h[:,dim]/3.,-D3[:,:3]@h[:,dim]/3.]
   radius=CONFIG['maximum_coordinate_correction_m'][dim];lo=q[:,dim]-radius;hi=q[:,dim]+radius
   if dim==1:
    bounds=road.bounds(target[:,0]);lo=np.maximum(lo,bounds[-1]+.9-origin[1]);hi=np.minimum(hi,bounds[0]-.9-origin[1])
   if np.any(lo>=hi):audit['infeasible']+=1;failed=True;break
   fit=lsq_linear(A,rhs,bounds=(lo,hi),method='bvls',tol=1e-7,max_iter=100)
   if not fit.success or not np.isfinite(fit.x).all():audit['solver_failed']+=1;failed=True;break
   candidates.append(fit.x)
  if failed:continue
  xy=np.stack(candidates,axis=1)+origin;shift=np.linalg.norm(xy-target,axis=1)
  out[:2,agent,cut:cut+n]=((xy-LOW[:2])/(HIGH-LOW)[:2]).T

  delta=np.diff(np.vstack([origin,xy]),axis=0);speed=np.linalg.norm(delta,axis=1)/.1
  heading=np.arctan2(delta[:,1],delta[:,0]);active=speed>.6
  out[2,agent,cut:cut+n][active]=(heading[active]-LOW[2])/(HIGH[2]-LOW[2])
  audit['optimized']+=1;audit['max_correction_m']=max(audit['max_correction_m'],float(shift.max()));audit['correction_sum_m']+=float(shift.sum());audit['points']+=n
 return out,audit

class OptimizedDiffusion:
 def __init__(self,diffusion,mode,road):
  self.diffusion,self.mode,self.road=diffusion,mode,road;self.first_raw_sha256=None;self.audits=[]
 def sample(self,model,shape,risk,history,mask,initial,**kwargs):
  raw=self.diffusion.sample(model,shape,risk,history,mask,initial,**kwargs);tracks=unflatten_agents(raw).cpu().numpy()
  if self.first_raw_sha256 is None:self.first_raw_sha256=hashlib.sha256(tracks.tobytes()).hexdigest()
  result=[]
  for x,cut in zip(tracks,mask[:,0].sum(-1).to(torch.int64).cpu().tolist()):
   y=smooth_future(x,cut,'joint')
   if self.mode=='optimized':
    y,audit=optimize_future(y,cut,self.road)
    y,road_audit=project_future_to_road(y,cut,self.road)
    self.audits.append(dict(history_length=cut,**audit,road_projection=road_audit,road_projection_config=ROAD_PROJECTION_CONFIG))
   result.append(y)
  return flatten_agents(torch.from_numpy(np.stack(result)).to(raw.device,dtype=raw.dtype))
