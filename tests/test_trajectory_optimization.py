import unittest
import numpy as np
from hdg.trajectory_optimization import optimize_future
from hdg.representation import LOW,HIGH
class FlatRoad:
 def bounds(self,x):return np.array([np.full_like(x,955.),np.full_like(x,945.),np.full_like(x,935.)])
class OptimizationTests(unittest.TestCase):
 def data(self):
  x=np.zeros((3,1,30));x[0,0]=(1010.+np.arange(30)*.5-LOW[0])/(HIGH[0]-LOW[0]);x[1,0]=.5;x[2,0]=.5;return x
 def test_linear_preserved(self):
  x=self.data();y,a=optimize_future(x,5,FlatRoad());np.testing.assert_allclose(y,x,atol=1e-8);self.assertEqual(a['optimized'],1)
 def test_correction_bounded_and_history_fixed(self):
  x=self.data();x[0,0,5:]+=.01;y,a=optimize_future(x,5,FlatRoad());np.testing.assert_array_equal(y[...,:5],x[...,:5]);self.assertLessEqual(a['max_correction_m'],np.sqrt(10)+1e-6)
  old=abs(x[0,0,5]-2*x[0,0,4]+x[0,0,3]);new=abs(y[0,0,5]-2*y[0,0,4]+y[0,0,3]);self.assertLess(new,old)
 def test_missing_history_skipped(self):
  x=self.data();x[:,:,4]=-1;y,a=optimize_future(x,5,FlatRoad());np.testing.assert_array_equal(y,x);self.assertEqual(a['skipped'],1)
