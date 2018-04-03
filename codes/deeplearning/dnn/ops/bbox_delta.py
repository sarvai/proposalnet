import os
import tensorflow as tf
from tensorflow.python.framework import ops

fname = os.path.join(os.path.dirname(__file__),'build','bbox_delta.so')
_bbox_delta_module = tf.load_op_library(fname)
bbox_delta_inv = _bbox_delta_module.bbox_delta_inv

def inv_op( rois, deltas ):
	return bbox_delta_inv( rois, deltas )

@ops.RegisterShape("BboxDeltaInv")
def _bbox_deta_inv_shape( op ):
    # Shape of the proposal_decode function
    return [ op.inputs[0].get_shape() ]
