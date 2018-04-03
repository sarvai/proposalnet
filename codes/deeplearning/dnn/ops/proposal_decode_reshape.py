import os
import tensorflow as tf
from tensorflow.python.framework import ops

fname = os.path.join(os.path.dirname(__file__),'build','proposal_decode_reshape.so')
_proposal_decode_module = tf.load_op_library(fname)
proposal_decode_reshape = _proposal_decode_module.proposal_decode_reshape

def op( bottom, spatial_scale ):
    return proposal_decode_reshape( bottom, spatial_scale=spatial_scale )

@ops.RegisterShape("ProposalDecodeReshape")
def _proposal_decode_reshape( op ):
    # Shape of the proposal_decode function
    nbatches, height, width, nc = op.inputs[0].get_shape()
    output_shape = tf.TensorShape([nbatches*height*width,5])
    return [ output_shape ]
