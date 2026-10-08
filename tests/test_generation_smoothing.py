import unittest
import numpy as np
from hdg.generation_smoothing import smooth_future

class SmoothingTests(unittest.TestCase):
    def test_history_fixed_and_joint_reads_history(self):
        x=np.full((3,1,30),.5,dtype=np.float32);x[:2,:,:10]=0
        joint=smooth_future(x,10,'joint');separate=smooth_future(x,10,'separate')
        for y in [joint,separate]:
            np.testing.assert_array_equal(y[...,:10],x[...,:10]);np.testing.assert_array_equal(y[2],x[2])
        self.assertLess(joint[0,0,10],separate[0,0,10]);self.assertEqual(separate[0,0,10],.5)
    def test_separate_does_not_read_history(self):
        x=np.full((3,1,30),.5);y=x.copy();y[:2,:,:10]=.9
        np.testing.assert_array_equal(smooth_future(x,10,'separate')[...,10:],smooth_future(y,10,'separate')[...,10:])
    def test_no_padding_bridge(self):
        x=np.full((3,1,30),.5);x[:,:,12:15]=-1;y=x.copy();y[:2,:,:12]=.9
        for mode in ['joint','separate']:
            a=smooth_future(x,10,mode);b=smooth_future(y,10,mode)
            np.testing.assert_array_equal(a[:,:,12:15],x[:,:,12:15]);np.testing.assert_array_equal(a[:,:,15:],b[:,:,15:])
