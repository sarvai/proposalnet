import os
import tensorflow as tf
from tensorflow.python.framework import ops

fname = os.path.join(os.path.dirname(__file__),'build','proposal_decode.so')
_proposal_decode_module = tf.load_op_library(fname)
proposal_decode = _proposal_decode_module.proposal_decode

def op( bottom, spatial_scale ):
    return proposal_decode( bottom, spatial_scale=spatial_scale )

@ops.RegisterShape("ProposalDecode")
def _roi_pool_shape( op ):
    # Shape of the proposal_decode function
    return [ op.inputs[0].get_shape() ]
