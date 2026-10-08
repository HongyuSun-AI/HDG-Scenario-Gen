import unittest
import numpy as np
from hdg.closed_loop import track_target

class TrackingTests(unittest.TestCase):
    def test_constant_speed_reference(self):
        state=np.array([1010.,945.,0.]);speed=10.
        for t in range(100):
            previous=np.array([1010.+t,945.,0.]);target=previous+np.array([1.,0.,0.])
            state,speed=track_target(state,speed,target,previous_target=previous)
        np.testing.assert_allclose(state,[1110.,945.,0.],atol=1e-6)
        self.assertAlmostEqual(speed,10.)

    def test_persistent_position_error_cannot_cause_runaway(self):
        state=np.array([1010.,945.,0.]);speed=2.
        for _ in range(140):
            target=state+np.array([20.,0.,0.]);previous=target-np.array([.2,0.,0.])
            state,speed=track_target(state,speed,target,previous_target=previous)
            self.assertLessEqual(speed,4.000001)

    def test_high_speed_lateral_acceleration_bound(self):
        state=np.array([1010.,945.,0.]);speed=30.
        for _ in range(100):
            target=state+np.array([0.,20.,0.]);target[2]=2.
            nxt,ns=track_target(state,speed,target,previous_target=target)
            distance=.5*(speed+ns)*.1
            angle=(nxt[2]-state[2]+np.pi)%(2*np.pi)-np.pi
            self.assertLessEqual(abs(angle)/max(distance,1e-9)*max(speed,ns)**2,3.000001)
            self.assertLessEqual(ns-speed,.300001)
            state,speed=nxt,ns

    def test_invalid_target_brakes_without_nan(self):
        state,speed=track_target(np.array([1010.,945.,0.]),2.,np.full(3,np.nan))
        self.assertTrue(np.isfinite(state).all());self.assertAlmostEqual(speed,1.5)
