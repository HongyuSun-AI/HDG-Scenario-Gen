import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from hdg.collision_attribution import (VehicleGeometry, CollisionType, classify_collision,
    box_intersects, in_common_lane, is_at_fault, CollisionRecorder)
from hdg.metrics import corners, pool_episodes


class StraightRoad:
    xmin, xmax = -100., 100.

    def bounds(self, x):
        return np.broadcast_to(np.array([6., 2., -2., -6.])[:, None], (4, len(x)))


class AttributionTests(unittest.TestCase):
    def test_decision_tree_and_stopped_priority(self):
        ego = np.array([0., 0., 0.])
        self.assertEqual(classify_collision(ego, np.array([4., 0., 0.]), 0, 0), CollisionType.STOPPED_EGO_COLLISION)
        self.assertEqual(classify_collision(ego, np.array([4., 0., 0.]), 5, 0), CollisionType.STOPPED_TRACK_COLLISION)
        self.assertEqual(classify_collision(ego, np.array([4., 0., 0.]), 5, 5), CollisionType.ACTIVE_FRONT_COLLISION)
        self.assertEqual(classify_collision(ego, np.array([-4., 0., 0.]), 5, 5), CollisionType.ACTIVE_REAR_COLLISION)
        self.assertEqual(classify_collision(ego, np.array([-.5, 1.5, 0.]), 5, 5), CollisionType.ACTIVE_LATERAL_COLLISION)
        self.assertFalse(is_at_fault(CollisionType.STOPPED_EGO_COLLISION, False))
        self.assertFalse(is_at_fault(CollisionType.ACTIVE_REAR_COLLISION, False))
        self.assertTrue(is_at_fault(CollisionType.STOPPED_TRACK_COLLISION, None))
        self.assertFalse(is_at_fault(CollisionType.ACTIVE_LATERAL_COLLISION, True))
        self.assertTrue(is_at_fault(CollisionType.ACTIVE_LATERAL_COLLISION, False))
        self.assertIsNone(is_at_fault(CollisionType.ACTIVE_LATERAL_COLLISION, None))

    def test_corner_lane_checks(self):
        road = StraightRoad()
        self.assertTrue(in_common_lane(np.array([0., 0., 0.]), road))
        self.assertFalse(in_common_lane(np.array([0., 2., 0.]), road))
        self.assertFalse(in_common_lane(np.array([0., 6., 0.]), road))
        self.assertIsNone(in_common_lane(np.array([99., 0., 0.]), road))

    def test_track_dedup_and_pooled_counts(self):
        recorder = CollisionRecorder(0, StraightRoad())
        states = np.array([[0., 0., 0.], [4., 0., 0.]])
        recorder.observe(states, np.ones(2, bool), [5., 0.], 2)
        recorder.observe(states, np.ones(2, bool), [0., 0.], 3)
        result = recorder.result()
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['events'][0]['frame'], 2)
        self.assertTrue(result['at_fault'])
        records = [dict(collision=True, at_fault=True, verification={'verified':True}),
                   dict(collision=True, at_fault=False, verification={'verified':False})]
        records += [dict(collision=False, at_fault=False) for _ in range(8)]
        pooled = pool_episodes(records)
        for k, value in [('collision_pct',20),('acr_pct',10),('varc_pct',50),('varaf_pct',100),('vafr_pct',10)]:
            self.assertEqual(pooled[k], value)
        records[0]['at_fault'] = None
        self.assertIsNone(pool_episodes(records)['acr_pct'])
        self.assertEqual(pool_episodes(records)['collision_pct'],20)
        self.assertIsNone(pool_episodes([])['collision_pct'])

    def test_osm_coverage_is_not_training_crop(self):
        road = StraightRoad()
        road.lines = [np.array([[-110., y], [110., y]]) for y in [6.,2.,-2.,-6.]]
        self.assertTrue(in_common_lane(np.array([101., 0., 0.]), road))
        self.assertFalse(in_common_lane(np.array([101., 2., 0.]), road))
        self.assertIsNone(in_common_lane(np.array([109., 0., 0.]), road))

    def test_against_pinned_upstream_classifier(self):


        from shapely.geometry import Polygon, LineString
        source = Path('references/nuplan/nuplan/planning/metrics/evaluation_metrics/common/no_ego_at_fault_collisions.py')
        tree = ast.parse(source.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_get_collision_type')
        def behind(ego, track):
            delta = np.array([track.x - ego.x, track.y - ego.y])
            cosine = np.dot([np.cos(ego.heading), np.sin(ego.heading)], delta / np.linalg.norm(delta))
            return np.arccos(np.clip(cosine, -1, 1)) > np.deg2rad(150)
        scope = dict(EgoState=object, TrackedObject=object, CollisionType=CollisionType,
                     LineString=LineString, is_agent_behind=behind,
                     is_track_stopped=lambda x: x.speed <= .05)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
        rng = np.random.default_rng(7)
        geometry = VehicleGeometry()
        tested = 0
        for _ in range(300):
            ego = np.r_[rng.uniform(-10, 10, 2), rng.uniform(-np.pi, np.pi)]
            other = ego + rng.uniform([-5,-3,-1], [5,3,1])
            if not box_intersects(ego, other, geometry, geometry):
                continue
            a, b = rng.choice([0., .049, .051, 5.], size=2)
            rear = ego[:2] - geometry.rear_axle_to_center * np.array([np.cos(ego[2]), np.sin(ego[2])])
            ego_state = SimpleNamespace(dynamic_car_state=SimpleNamespace(speed=a),
                rear_axle=SimpleNamespace(x=rear[0], y=rear[1], heading=ego[2]),
                car_footprint=SimpleNamespace(oriented_box=SimpleNamespace(geometry=Polygon(corners(ego)[[0,3,2,1]]))))
            track = SimpleNamespace(speed=b, box=SimpleNamespace(geometry=Polygon(corners(other)),
                center=SimpleNamespace(x=other[0], y=other[1], heading=other[2])))
            self.assertEqual(classify_collision(ego, other, a, b), scope['_get_collision_type'](ego_state, track))
            tested += 1
        self.assertGreater(tested, 100)


if __name__ == '__main__':
    unittest.main()
