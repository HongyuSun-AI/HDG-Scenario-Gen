import unittest
import numpy as np
from hdg.sv_stable import StableSVController
from hdg.road import MergeRoad
from hdg.representation import normalize
from hdg.sv_diagnostics import overlap_matrix
from hdg.metrics import overlap
class StableTests(unittest.TestCase):
 def test_matrix_matches_sat(self):
  rng=np.random.default_rng(7);s=rng.normal(size=(12,3))*[5,2,1]
  m=overlap_matrix(s)
  for i in range(12):
   for j in range(12):self.assertEqual(bool(m[i,j]),overlap(s[i],s[j]))
 def test_alternating_generated_yaw_does_not_steer(self):
  r=MergeRoad();s=np.array([[1020,r.center(1,1020),0.],[1050,r.center(0,1050),0.]])
  v=np.array([True,True]);p=np.repeat(s[:,None],20,axis=1);p[:,:,0]+=np.arange(20)*.3;p[0,:,2]=np.where(np.arange(20)%2,3.14,-3.14)
  plan=normalize(p,np.ones((2,20),bool));c=StableSVController(r,2)
  ns,sp,a=c.propose(s,v,np.array([3.,3.]),plan,1,1)
  self.assertLess(abs(ns[0,2]),.01);self.assertLessEqual(abs(sp[0]-3),.031)
 def test_predictive_yield_catches_side_convergence(self):
  r=MergeRoad();s=np.array([[1050.,941.,.3],[1050.,944.,-.3],[1010.,945.,0.]])
  p=np.repeat(s[:,None],20,axis=1);p[:,:,0]+=np.arange(20)*.3
  plan=normalize(p,np.ones((3,20),bool));c=StableSVController(r,3)
  ns,sp,a=c.propose(s,np.ones(3,bool),np.array([3.,3.,1.]),plan,1,2)
  self.assertTrue(any(x['kind']=='sv_predictive_yield' for x in a))
 def test_memory_survives_window_replan(self):
  r=MergeRoad();s=np.array([[1020.,r.center(1,1020),0.],[1100.,r.center(0,1100),0.]])
  p=np.repeat(s[:,None],30,axis=1);p[:,:,0]+=np.arange(30)*.5
  plan=normalize(p,np.ones((2,30),bool));c=StableSVController(r,2);speed=np.array([1.,1.]);acc=[]
  for t in range(1,12):
   ns,sp,a=c.propose(s,np.ones(2,bool),speed,plan,t,1);acc.append((sp[0]-speed[0])/.1);s=ns;speed=sp
  self.assertLessEqual(np.max(np.abs(np.diff(acc))),.30001)
if __name__=='__main__':unittest.main()
