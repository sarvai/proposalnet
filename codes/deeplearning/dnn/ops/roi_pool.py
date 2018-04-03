import os
import tensorflow as tf
from tensorflow.python.framework import ops

fname = os.path.join(os.path.dirname(__file__),'build','roi_pool.so')

_roi_pool_module = tf.load_op_library(fname)
roi_pool = _roi_pool_module.roi_pool
roi_pool_grad = _roi_pool_module.roi_pool_grad

def op( input, rois, pooled_height, pooled_width, spatial_scale ):
    return roi_pool( input, rois, pooled_height=pooled_height, pooled_width=pooled_width, spatial_scale=spatial_scale )[0]

def grad_op( input, argmax, grad ):
    return roi_pool_grad( input, argmax, grad )

@ops.RegisterShape("RoiPool")
def _roi_pool_shape( op ):
    # Shape of the roi_pool function
    dims_input = op.inputs[0].get_shape().as_list()
    nchannels = dims_input[3]
    dims_rois = op.inputs[1].get_shape().as_list()
    nrois = dims_rois[0]

    pooled_height = op.get_attr('pooled_height')
    pooled_width = op.get_attr('pooled_width')

    output_shape = tf.TensorShape([nrois,pooled_height,pooled_width,nchannels])

    return [ output_shape, output_shape ]

@ops.RegisterGradient("RoiPool")
def _roi_pool_grad( op, grad, _ ):
    # Gradient.
    input = op.inputs[0]
    argmax = op.outputs[1]
    input_grad = roi_pool_grad( input, argmax, grad )
    return [ input_grad, None ]
