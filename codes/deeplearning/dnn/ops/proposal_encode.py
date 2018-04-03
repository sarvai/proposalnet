import os
import tensorflow as tf
from tensorflow.python.framework import ops

fname = os.path.join(os.path.dirname(__file__),'build','proposal_encode.so')
_proposal_encode_module = tf.load_op_library(fname)
proposal_encode = _proposal_encode_module.proposal_encode

def op( feat_map, gtboxes, gtlabels, gtbatches, shapes, spatial_scale ):
    return proposal_encode( feat_map, gtboxes, 
                    gtlabels, gtbatches, shapes, 
                    spatial_scale=spatial_scale )

@ops.RegisterShape("ProposalEncode")
def _proposal_encode_shape( op ):
    # Shape of the proposal_encode tensors
    dims_input = op.inputs[0].get_shape().as_list()

    n,h,w,c = dims_input

    labels_shape = tf.TensorShape([n,h,w,1])
    targets_shape = tf.TensorShape([n,h,w,4])


    return [ labels_shape, targets_shape ]
