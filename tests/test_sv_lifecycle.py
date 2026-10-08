import unittest
import numpy as np
import torch
from hdg.batched_loop import run_batch
from hdg.sv_lifecycle import estimate_unknown_speeds
from hdg.representation import normalize
from hdg.road import MergeRoad
from hdg.closed_loop import ReactivePlanner
from tests.test_batched_experiment import TinyModel,RecordingDiffusion

class LifecycleTests(unittest.TestCase):
 def test_padding_removes_actor_and_does_not_respawn(self):
  class D(RecordingDiffusion):
   def sample(self,*args,**kw):
    x=super().sample(*args,**kw)

    x[:,[3,4,5],1:11]=-1
    return x
  r=MergeRoad();s=np.array([[1010.,r.center(1,1010.),0.],[1060.,r.center(0,1060.),0.]])
  es,_=run_batch(TinyModel(),D(),np.zeros((3,28,3)),normalize(s,np.ones(2,bool)),[1.,0.],0,[1130.,945.,0.],lambda:ReactivePlanner(r),batch_size=2,horizon=21)
  for e in es:
   self.assertTrue((e['tracks'][:,1,1:]==-1).all());self.assertEqual(e['lifecycle_events'][0]['reason'],'generated_padding')
 def test_malformed_is_not_absence(self):
  class D(RecordingDiffusion):
   def sample(self,*args,**kw):
    x=super().sample(*args,**kw);x[:,[3,4,5],1:]=-.5;return x
  r=MergeRoad();s=np.array([[1010.,r.center(1,1010.),0.],[1060.,r.center(0,1060.),0.]])
  es,_=run_batch(TinyModel(),D(),np.zeros((3,28,3)),normalize(s,np.ones(2,bool)),[1.,0.],0,[1130.,945.,0.],lambda:ReactivePlanner(r),batch_size=1,horizon=21)
  self.assertFalse(es[0]['lifecycle_events']);self.assertTrue((es[0]['tracks'][:,1]>=0).all())
 def test_unknown_speed_is_distinct_from_known_stop(self):
  r=MergeRoad();s=np.array([[1010.,r.center(1,1010.),0.],[1020.,r.center(1,1020.),0.],[1015.,r.center(0,1015.),0.]])
  v,a=estimate_unknown_speeds(s,np.ones(3,bool),np.array([0.,15.,0.]),np.array([False,True,True]),r)
  self.assertEqual(v[0],15.);self.assertEqual(v[2],0.);self.assertEqual(len(a),1)
