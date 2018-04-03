import os
import tensorflow as tf
from tensorflow.python.framework import ops

fname = os.path.join(os.path.dirname(__file__),'build','frcnn_prepare.so')
_frcnn_prepare_module = tf.load_op_library(fname)
frcnn_prepare = _frcnn_prepare_module.frcnn_prepare

def op( gtboxes, gtlabels, gtbatches, rois, roilabels, roibatches, overlaps, nclasses, fg_min_overlap, bg_max_overlap ):
    return frcnn_prepare( gtboxes, gtlabels, gtbatches,
                            rois, roilabels, roibatches,
                            overlaps,
                            nclasses=nclasses, fg_min_overlap=fg_min_overlap,
                            bg_max_overlap=bg_max_overlap )

@ops.RegisterShape("FrcnnPrepare")
def _frcnn_prepare_shape( op ):
    # Shape of the proposal_encode tensors
    nclasses = op.get_attr("nclasses")
    nrois = op.inputs[3].get_shape()[0]

    labels_shape = tf.TensorShape([nrois,nclasses])
    deltas_shape = tf.TensorShape([nrois,4*nclasses])

    return [ labels_shape, deltas_shape ]
