import unittest
import numpy as np
from hdg.left_lanechange_experiment import lane_change
from hdg.representation import LOW,HIGH
class FlatRoad:
 def bounds(self,x):return np.array([np.full_like(x,y) for y in [950.,946.,942.,938.]])
class LaneChangeAcceptance(unittest.TestCase):
 def episode(self,y):
  s=np.array([np.arange(len(y))*.2+1050,np.array(y),np.zeros(len(y))]);x=((s-LOW[:,None])/(HIGH-LOW)[:,None])[:,None,:]
  return dict(tracks=x,ego=0,collision_time=None,arrival_time=None,exit_time=None)
 def test_actual_full_body_stability(self):
  e=self.episode([940]+[944]*8);r=lane_change(e,FlatRoad());self.assertTrue(r['completed']);self.assertEqual(r['events'][0]['from_lane'],2)
 def test_straddling_start_is_not_lane_change(self):
  e=self.episode([942]+[944]*8);self.assertFalse(lane_change(e,FlatRoad())['completed'])
 def test_termination_before_confirmation(self):
  e=self.episode([940]+[944]*8);e['collision_time']=3;self.assertFalse(lane_change(e,FlatRoad())['completed'])

class ReferenceIntentSelection(unittest.TestCase):
 def test_source_risk_and_reference_intent_not_ego_risk(self):
  import torch
  from hdg.congestion_lanechange_experiment import candidates
  y=[940.]*4+[944.]*16
  s=np.array([1050.+np.arange(20)*.2,y,np.zeros(20)])
  x=((s-LOW[:,None])/(HIGH-LOW)[:,None])[:,None,:]
  data=dict(tracks=torch.tensor(x[None],dtype=torch.float32),continuous_risk=torch.tensor([.95]),agent_risk=torch.tensor([[.01]]),source_indices=torch.tensor([123]),order=torch.tensor([[0]]))
  result=candidates(data,FlatRoad(),7)
  self.assertEqual(len(result),1);self.assertEqual(result[0]['source_index'],123)
  data['continuous_risk']=torch.tensor([.89]);self.assertFalse(candidates(data,FlatRoad(),7))
