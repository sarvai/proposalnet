import os
import tensorflow as tf
from tensorflow.python.framework import ops

fname = os.path.join(os.path.dirname(__file__),'build','overlaps.so')
_overlaps_module = tf.load_op_library(fname)
overlaps = _overlaps_module.overlaps

def op( gtboxes, gtlabels, gtbatches, rois, roilabels, roibatches ):
    return overlaps( gtboxes, gtlabels, gtbatches, rois, roilabels, roibatches )

@ops.RegisterShape("Overlaps")
def _overlaps_shape( op ):
    # Shape of the proposal_encode tensors
    ngtboxes = op.inputs[0].get_shape()[0]
    nrois = op.inputs[3].get_shape()[0]

    overlaps_shape = tf.TensorShape([nrois,ngtboxes])

    return [ overlaps_shape ]
