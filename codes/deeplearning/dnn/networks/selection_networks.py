import tensorflow as tf
import tensorflow.contrib.slim as slim
from .network import network
from ..py_ops import common as ops_common

class flat( network ):
    def apply( self, net ):
        net = slim.max_pool2d(net, [2, 2], scope='pool5')
        net = slim.flatten( net )
        return net

class roipool( network ):
    def apply( self, net, roiboxes, roibatches ):
        base_size = self._dnn_cfg.COMMON.BASE_SIZE
        pool_height = self._dnn_cfg.FRCNN.POOL_HEIGHT
        pool_width = self._dnn_cfg.FRCNN.POOL_WIDTH

        roiboxes = tf.py_func( ops_common.prepare_rois, [ roiboxes, tf.shape( net ), base_size ], tf.float32 )
        roibatches = tf.reshape( roibatches, [-1] )
        net = tf.image.crop_and_resize( net, roiboxes, roibatches, [pool_height,pool_width] )
        net = slim.flatten( net )
        return net
