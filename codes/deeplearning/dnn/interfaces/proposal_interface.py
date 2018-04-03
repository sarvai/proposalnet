import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .interface import interface
from ..networks import networks

from ..ops import proposal_encode, proposal_decode
from ..py_ops import common as common_ops

class proposal_interface( interface ):
    def __init__( self, mcfg, dnn_cfg ):
        super().__init__( mcfg, dnn_cfg )
        self._sections['feat'] = networks.feature[ self._feat_name ]( self._dnn_cfg, prefix=self._prefix )
        self._sections['proposal'] = networks.proposal( self._dnn_cfg, prefix=self._prefix )
        self._nclasses = mcfg['nclasses']
        self._batchsize = mcfg['batch_size']

    def _init_end2end( self, add_gt_to_rois=False ):
        self._nets = {}
        self._outputs = {}

        self._nets['feat'] = self._sections['feat'].apply( self._inputs['data'] )['feat16']

        self._nets['proposal'] = self._sections['proposal'].apply( self._nets['feat']['net'], nclasses=self._nclasses )

        self._outputs['proposal'] = self._sections['proposal'].post_process( self._inputs, self._nets['proposal'],
                                                                       self._nclasses, self._batchsize,
                                                                       spatial_scale=self._nets['feat']['scale'] )

    def _train_end2end( self, stage=1 ):
        feat_variables = self._build_variable_list( self._sections['feat'].names )
        proposal_losses, proposal_var_names = self._sections['proposal'].loss( self._inputs, self._nets['feat']['net'],
                                                                               self._nets['proposal'], self._nclasses,
                                                                               spatial_scale=self._nets['feat']['scale'] )

        var_names = self._sections['feat'].names + proposal_var_names
        variables = self._build_variable_list( var_names )

        return proposal_losses, self._inputs, variables

    def train_loss_end2end( self, stage=1 ):
        self._inputs = {}

        self._inputs['data'] = tf.placeholder( tf.float32, shape=[ None, None, None, 3 ] )
        self._inputs['shapes'] = tf.placeholder( tf.float32, shape=[ None, 2 ] )
        self._inputs['batch_labels'] = tf.placeholder( tf.float32, shape=[ None, 1 ] )
        self._inputs['gtboxes'] = tf.placeholder( tf.float32, shape=[ None, 4 ] )
        self._inputs['gtlabels'] = tf.placeholder( tf.float32, shape=[ None, 1 ] )
        self._inputs['gtbatches'] = tf.placeholder( tf.int32, shape=[ None, 1 ] )
        self._inputs['dropout_prob'] = tf.placeholder_with_default(0.5, shape=())

        for name, sec in self._sections.items() :
            sec.reset()

        self._init_end2end( add_gt_to_rois=self._dnn_cfg.FRCNN.TRAIN.ADD_GTBOXES )

        return self._train_end2end( stage=stage )
        #losses, variables = self._train_end2end( stage=stage )
        #return losses, inputs, variables

    def test_out( self ):
        self._inputs = {}

        self._inputs['data'] = tf.placeholder( tf.float32, shape=[ None, None, None, 3 ] )
        self._inputs['shapes'] = tf.placeholder( tf.float32, shape=[ None, 2 ] )
        self._inputs['batch_labels'] = tf.placeholder( tf.float32, shape=[ None, 1 ] )
        self._inputs['gtboxes'] = tf.placeholder( tf.float32, shape=[ None, 4 ] )
        self._inputs['gtlabels'] = tf.placeholder( tf.float32, shape=[ None, 1 ] )
        self._inputs['gtbatches'] = tf.placeholder( tf.int32, shape=[ None, 1 ] )
        self._inputs['dropout_prob'] = tf.placeholder_with_default(0.5, shape=())

        for name, sec in self._sections.items() :
            sec.reset()

        self._init_end2end( add_gt_to_rois=self._dnn_cfg.FRCNN.TRAIN.ADD_GTBOXES )

        losses, inputs, variables = self._train_end2end()

        out = {}
        out['inputs'] = inputs
        out['outputs'] = losses

        return out

        #return self._train_end2end( stage=stage )


    def deploy( self, frcnn_nms=True ):
        for name, sec in self._sections.items() :
            sec.reset()

        self._inputs = {}
        self._variables = {}

        self._inputs['data'] = tf.placeholder( tf.float32, shape=[ None, None, None, 3 ] )
        self._inputs['shapes'] = tf.placeholder( tf.float32, shape=[ None, 2 ] )
        self._inputs['dropout_prob'] = tf.placeholder_with_default(1.0, shape=())

        self._init_end2end( add_gt_to_rois=False )

        out = {}
        out['inputs'] = self._inputs
        out['outputs'] = self._outputs['proposal']

        return out
