import unittest
import numpy as np
from hdg.road import MergeRoad
from hdg.pdm_planner import PDMClosedAdapter, PDMConfig
from hdg.pdm_replay import FixedFutureTrajectory
from hdg.closed_loop import verify_fixed_replay
from hdg.representation import normalize


class PDMTests(unittest.TestCase):
    def setUp(self):
        self.road = MergeRoad()
        self.states = np.array([[1010.,self.road.center(1,1010.),0.],
                                [1028.,self.road.center(0,1028.),0.]])
        self.goal = np.array([1130.,self.road.center(1,1130.),0.])

    def planner(self):
        p = PDMClosedAdapter(self.road)
        p.reset(self.states, 0, self.goal)
        p.set_initial_speeds([10.,0.])
        return p

    def test_immutable_time_aligned_replay_birth_exit_and_tail(self):
        x = np.repeat(self.states[None], 5, axis=0)
        x[:,1,0] += np.arange(5)
        x[0,1] = np.nan
        replay = FixedFutureTrajectory(x,0,.1,2.)
        x[1,1,0] = 9999
        self.assertFalse(replay.at(2.)[1][1])
        self.assertAlmostEqual(replay.at(2.1)[0][1,0],1029)
        self.assertFalse(replay.valid[:,0].any())
        self.assertTrue(replay.at(3.)[2])
        np.testing.assert_array_equal(replay.velocity(10,1), [0.,0.])
        with self.assertRaises(ValueError): replay.at(2.05)
        with self.assertRaises(ValueError): replay.states[1,1,0] = 1
        x[-1,1] = np.nan
        ended = FixedFutureTrajectory(x,0,.1,2.)
        self.assertFalse(ended.at(3.)[1][1])

    def test_future_changes_official_candidate_scores_and_execution(self):
        future = np.repeat(self.states[None],140,axis=0)
        future[:,1,1] += (self.road.center(1,1028.)-self.states[1,1])*np.clip(np.arange(1,141)/10.,0,1)
        ordinary, oracle = self.planner(), self.planner()
        normal_state, normal_speed = ordinary.step(self.states,np.ones(2,bool),10.,.1)
        oracle_state, oracle_speed = oracle.step(self.states,np.ones(2,bool),10.,.1,future)
        self.assertLess(oracle_speed, normal_speed-.01)
        self.assertFalse(np.array_equal(ordinary.core._scorer._multi_metrics,oracle.core._scorer._multi_metrics))
        self.assertFalse(ordinary.audits[-1]['known_future_used'])
        self.assertTrue(oracle.audits[-1]['known_future_used'])
        self.assertEqual(oracle.audits[-1]['padded_forecast_frames'],0)
        observation = oracle.core._observation
        token = 'hdg_agent_1'
        polygon = observation[10][token]
        self.assertAlmostEqual(polygon.centroid.y, future[9,1,1])
        expected_velocity = (future[10,1,:2]-future[8,1,:2])/.2
        self.assertAlmostEqual(observation.unique_objects[token].velocity.magnitude(),np.linalg.norm(expected_velocity))

        future[:,0] = [1090.,945.,2.]
        other_oracle = self.planner()
        again, speed = other_oracle.step(self.states,np.ones(2,bool),10.,.1,future)
        np.testing.assert_allclose(again, oracle_state, atol=1e-10)
        self.assertAlmostEqual(speed,oracle_speed)

    def test_future_rotation_reaches_occupancy_geometry(self):
        future = np.repeat(self.states[None],100,axis=0)
        future[:,1,2] = np.pi/2
        p = self.planner()
        p.step(self.states,np.ones(2,bool),10.,.1,future)
        observation = p.core._observation
        bounds = observation[1]['hdg_agent_1'].bounds
        self.assertAlmostEqual(bounds[2]-bounds[0],1.8)
        self.assertAlmostEqual(bounds[3]-bounds[1],4.5)
        self.assertAlmostEqual(observation.unique_objects['hdg_agent_1'].box.center.heading,np.pi/2)

    def test_verification_until_exit_keeps_surroundings_and_planner(self):
        horizon = 140
        states = np.repeat(self.states[:,None],horizon,axis=1)
        states[0,:,0] += np.arange(horizon)*.8
        states[1,:,0] = 1050.
        original = normalize(states,np.ones((2,horizon),bool)).transpose(2,0,1)
        episode = dict(tracks=original.copy(),ego=0,collision_time=5,
                       planner_config=self.planner().describe())
        p = PDMClosedAdapter(self.road)
        result = verify_fixed_replay(episode,[8.,0.],p,self.goal)
        self.assertEqual(len(result['ego_trajectory']),result['evaluated_frames'])
        self.assertLess(result['evaluated_frames'],140)
        self.assertEqual(result['exit_time'],result['evaluated_frames'])
        self.assertTrue((np.asarray(result['ego_trajectory'])[:,0] <= self.road.xmax).all())
        self.assertEqual(len(result['planner_audits']),result['exit_time'])
        self.assertTrue(all(a['known_future_used'] for a in result['planner_audits']))
        self.assertTrue(result['planner_audits'][-1]['padded_forecast_frames'] > 0)
        np.testing.assert_array_equal(episode['tracks'], original)
        self.assertTrue(np.isfinite(result['ego_trajectory']).all())
        wrong = PDMClosedAdapter(self.road,PDMConfig(speed_limit_mps=12.))
        with self.assertRaises(ValueError): verify_fixed_replay(episode,[8.,0.],wrong,self.goal)


if __name__ == '__main__':
    unittest.main()
