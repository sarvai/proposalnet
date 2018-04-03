import os
import tensorflow as tf
from tensorflow.python.framework import ops

fname = os.path.join(os.path.dirname(__file__),'build','smooth_L1.so')
_smooth_L1_module = tf.load_op_library(fname)

smooth_L1 = _smooth_L1_module.smooth_l1
smooth_L1_grad = _smooth_L1_module.smooth_l1_grad

def op( diffs, sigma=0.01 ):
	return smooth_L1( diffs, sigma )

def grad_op( diffs, top_grad, sigma=0.01 ):
	return smooth_L1_grad( diffs, top_grad, sigma )

@ops.RegisterShape("SmoothL1")
def _smooth_l1_loss_shape( op ):
	return [ op.inputs[0].get_shape() ]

@ops.RegisterGradient("SmoothL1")
def _smooth_l1_loss_grad( op, top_grad ):
	sigma = op.get_attr('sigma')
	diffs = op.inputs[0]
	bottom_grad = smooth_L1_grad( diffs, top_grad, sigma )
	return [ bottom_grad ]