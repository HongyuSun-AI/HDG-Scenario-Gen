import unittest
from unittest.mock import patch
import numpy as np
from hdg.closed_loop import ReactivePlanner,run_closed_loop,verify_fixed_replay,evaluation_end,reached_destination
from hdg.batched_loop import run_batch
from hdg.road import MergeRoad
from hdg.representation import normalize
from tests.test_batched_experiment import TinyModel,RecordingDiffusion

class ExitPlanner(ReactivePlanner):
    stop_at_destination=False
    def step(self,observed,valid,speed,dt,future_env=None):
        self.calls=getattr(self,'calls',0)+1
        state=observed[self.ego].copy();state[0]+=1
        return state,10.

class RegionExitTests(unittest.TestCase):
    def setUp(self):
        self.road=MergeRoad();self.s=np.array([[1138.,self.road.center(1,1138.),0.],[1100.,self.road.center(0,1100.),0.]])
        self.first=normalize(self.s,np.ones(2,bool));self.goal=self.s[0].copy();self.initial=np.zeros((3,28,3),np.float32)
    def test_batch_exit_no_reentry_or_post_exit_collision(self):
        planners=[]
        def factory():
            p=ExitPlanner(self.road);planners.append(p);return p
        episodes,_=run_batch(TinyModel(),RecordingDiffusion(),self.initial,self.first,[10.,0.],0,self.goal,factory,batch_size=2,horizon=21)
        for e,p in zip(episodes,planners):
            self.assertEqual(e['exit_time'],3);self.assertEqual(p.calls,3)
            self.assertFalse(e['collision']);self.assertIsNone(e['arrival_time'])
            self.assertTrue((e['tracks'][:,0,3:]==-1).all())
            self.assertEqual(evaluation_end(e,21),3)
    def test_single_exit_matches_batch(self):
        import torch
        p=ExitPlanner(self.road)

        d=RecordingDiffusion();original=d.sample
        d.sample=lambda *a,**kw:original(*a,**dict(kw,generator=torch.Generator().manual_seed(7)))
        e=run_closed_loop(TinyModel(),d,torch.tensor(self.initial)[None],self.first,[10.,0.],1.,0,p,self.goal,21)
        self.assertEqual(e['exit_time'],3);self.assertFalse(e['collision']);self.assertEqual(p.calls,3)
    def test_verification_stops_at_own_exit(self):
        s=np.repeat(self.s[:,None],21,axis=1)
        e=dict(tracks=normalize(s,np.ones((2,21),bool)).transpose(2,0,1),ego=0)
        p=ExitPlanner(self.road);v=verify_fixed_replay(e,[10.,0.],p,self.goal)
        self.assertEqual(v['exit_time'],3);self.assertEqual(v['evaluated_frames'],3);self.assertEqual(p.calls,3)
        self.assertFalse(v['collision']);self.assertEqual(len(v['ego_trajectory']),3)
    def test_ttc_excludes_exit_and_post_exit_frames(self):
        from hdg.night_experiment import summarize
        s=np.repeat(self.s[:,None],140,axis=1)
        e=dict(tracks=normalize(s,np.ones((2,140),bool)).transpose(2,0,1),ego=0,
               collision_time=None,arrival_time=None,exit_time=3,collision=False,at_fault=False,
               verified=False,boundaries=[],history_max_error=0.,invalid_generated_targets=0)
        with patch('hdg.night_experiment.minimum_ttc',return_value=None) as ttc:
            report=summarize([e],[e['tracks']])
        self.assertEqual(ttc.call_args[0][0].shape[1],3)
        self.assertEqual(report['counts']['collisions'],0)
        self.assertEqual(report['exited_episodes'],1)

    def test_pdm_route_not_truncated_at_recorded_endpoint(self):
        from hdg.pdm_planner import PDMClosedAdapter
        p=PDMClosedAdapter(self.road);p.reset(self.s,0,self.goal)
        self.assertFalse(reached_destination(p,self.goal,self.goal))
        self.assertTrue(all(l.baseline_path.discrete_path[-1].x>=self.road.xmax+199 for l in p.map_api.lanes.values()))
