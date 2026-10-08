import unittest
import numpy as np
import torch
from hdg.road import MergeRoad
from hdg.road_projection import project_future_to_road,center_limits
from hdg.representation import LOW,HIGH
from hdg.metrics import corners

class RoadProjectionTests(unittest.TestCase):
 def test_pi_heading_storage_roundtrip(self):
  class FlatRoad:
   def bounds(self,x):return np.array([np.full_like(x,955.),np.full_like(x,945.),np.full_like(x,936.)])
   def center(self,lane,x):return np.full_like(x,940.)
  s=np.array([1100-np.arange(20)*.5,np.full(20,935.),np.zeros(20)])
  x=((s-LOW[:,None])/(HIGH-LOW)[:,None])[:,None,:].astype(np.float32)
  y,a=project_future_to_road(x,0,FlatRoad())
  self.assertTrue(a['representable_heading_retry']);self.assertEqual(a['remaining_violations'],0)
  stored=np.clip(y[:,0].T,0,1)*(HIGH-LOW)+LOW
  lo,hi=center_limits(stored[:,0],stored[:,2],FlatRoad())
  self.assertTrue((stored[:,1]>=lo-1e-4).all());self.assertTrue((stored[:,1]<=hi+1e-4).all())
 def data(self):
  x=np.linspace(1090,1138,40);r=MergeRoad()

  y=r.center(1,x)-5
  return ((np.stack([x,y,np.zeros_like(x)])-LOW[:,None])/(HIGH-LOW)[:,None])[:,None,:]
 def check(self,y,cut):
  r=MergeRoad();states=y[:,0,cut:].T*(HIGH-LOW)+LOW
  for s in states:
   c=corners(s);b=r.bounds(c[:,0])
   self.assertTrue((c[:,1]>=b[-1]+.049).all())
   self.assertTrue((c[:,1]<=b[0]-.049).all())
 def test_large_departure_short_history(self):
  x=self.data();y,a=project_future_to_road(x,1,MergeRoad())
  np.testing.assert_array_equal(x[...,:1],y[...,:1]);self.check(y,1)
  self.assertGreater(a['max_lateral_correction_m'],1);self.assertEqual(a['remaining_violations'],0)
 def test_padding_and_later_segment(self):
  x=self.data();x[:,:,8:12]=-1
  y,a=project_future_to_road(x,1,MergeRoad())
  np.testing.assert_array_equal(x[:,:,8:12],y[:,:,8:12]);self.check(y[:,:,12:],0)
  self.assertEqual(a['repaired_segments'],2)
 def test_legal_lane_change_unchanged(self):
  r=MergeRoad();x=self.data();xx=x[0,0]*140+1000
  x[1,0]=(r.center(1,xx)+np.linspace(0,2,40)-935)/20
  x[2,0]=(np.arctan((r.center(1,xx+.1)-r.center(1,xx-.1))/.2)+3.14)/6.28
  y,a=project_future_to_road(x,1,r)
  np.testing.assert_array_equal(x,y);self.assertEqual(a['repaired_segments'],0)
 def test_rotated_vehicle(self):
  x=self.data();x[2,0]=(.8+3.14)/6.28
  y,a=project_future_to_road(x,1,MergeRoad());self.check(y,1)
 def test_generation_wrapper_applies_final_constraint(self):
  from hdg.trajectory_optimization import OptimizedDiffusion
  from hdg.representation import flatten_agents,unflatten_agents,history_mask
  x=self.data().astype(np.float32);raw=flatten_agents(torch.from_numpy(x[None]))
  class FixedDiffusion:
   def sample(self,*args,**kwargs):return raw.clone()
  wrapper=OptimizedDiffusion(FixedDiffusion(),'optimized',MergeRoad())
  result=wrapper.sample(None,raw.shape,None,raw,history_mask([1],40),None)
  y=unflatten_agents(result)[0].numpy()
  self.check(y,1);np.testing.assert_array_equal(y[...,:1],x[...,:1])
  self.assertGreater(wrapper.audits[0]['road_projection']['repaired_segments'],0)

if __name__=='__main__':unittest.main()
