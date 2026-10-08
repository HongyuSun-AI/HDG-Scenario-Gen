import unittest
import numpy as np
from hdg.closed_loop import track_target,TRACKING_CONFIG
from hdg.sv_front_guard import apply_front_guard,controller_metadata

class FrontGuardTests(unittest.TestCase):
 def setup_case(self,leader,speed=2.,leader_speed=0.):
  s=np.array([[1010.,945.,0.],leader],dtype=float);speeds=np.array([speed,leader_speed]);target=s[0]+[4.,0.,0.];previous=target-[speed*.1,0.,0.]
  proposal,v=track_target(s[0],speed,target,previous_target=previous)
  return s,speeds,proposal,v
 def test_unobstructed_and_adjacent_and_behind_are_exact(self):
  for leader in [[1018.,948.5,0.],[1007.,945.,0.],[1038.,945.,0.],[1016.,945.,np.pi/2]]:
   s,vs,p,v=self.setup_case(leader);out,nv,a=apply_front_guard(0,s,[True,True],vs,p,v)
   np.testing.assert_array_equal(out,p);self.assertEqual(nv,v);self.assertIsNone(a)
 def test_near_stopped_leader_brakes(self):
  s,vs,p,v=self.setup_case([1015.5,945.,0.]);out,nv,a=apply_front_guard(0,s,[True,True],vs,p,v)
  self.assertIsNotNone(a);self.assertLess(nv,vs[0]);self.assertGreaterEqual(nv,vs[0]-.400001)
 def test_moving_front_at_equal_speed_not_automatically_restricted(self):
  s,vs,p,v=self.setup_case([1017.,945.,0.],leader_speed=2.);out,nv,a=apply_front_guard(0,s,[True,True],vs,p,v)
  self.assertIsNone(a);self.assertEqual(nv,v)
 def test_existing_stronger_braking_preserved(self):
  s,vs,p,v=self.setup_case([1015.,945.,0.]);p,v=track_target(s[0],vs[0],np.full(3,np.nan))
  out,nv,a=apply_front_guard(0,s,[True,True],vs,p,v)
  np.testing.assert_array_equal(out,p);self.assertEqual(nv,v)
 def test_reference_chasing_stopped_vehicle_is_limited(self):
  s=np.array([[1010.,945.,0.],[1018.,945.,0.]]);vs=np.array([2.,0.]);interventions=0
  for _ in range(70):
   target=s[0]+[3.,0.,0.];p,v=track_target(s[0],vs[0],target,previous_target=target-[.2,0.,0.])
   p,v,a=apply_front_guard(0,s,[True,True],vs,p,v);s[0]=p;vs[0]=v;interventions+=a is not None
   self.assertGreater(s[1,0]-s[0,0],4.5)
  self.assertGreater(interventions,0)
 def test_highway_batch_is_unchanged(self):
  from hdg.batched_loop import run_batch
  from tests.test_batched_experiment import TinyModel,RecordingDiffusion,IndependentPlanner
  from hdg.road import MergeRoad
  from hdg.representation import normalize
  r=MergeRoad();s=np.array([[1010.,r.center(1,1010.),0.],[1060.,r.center(0,1060.),0.]])
  args=(TinyModel(),RecordingDiffusion(),np.zeros((3,28,3)),normalize(s,np.ones(2,bool)),[1.,0.],0,[1130.,945.,0.],lambda:IndependentPlanner(r))
  IndependentPlanner.count=0;a,_=run_batch(*args,batch_size=2,horizon=21)
  IndependentPlanner.count=0;b,_=run_batch(*args,batch_size=2,horizon=21,sv_controller='trajectory')
  for x,y in zip(a,b):np.testing.assert_array_equal(x['tracks'],y['tracks'])
  self.assertEqual(controller_metadata('trajectory',TRACKING_CONFIG),TRACKING_CONFIG)
