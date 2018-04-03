import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from .network import network

#custom ops
from ..ops import roi_pool, loss_ops, proposal_decode, proposal_encode, bbox_delta
from ..py_ops import common, proposal

def nan_check( mat, st ):
    if np.isnan( mat ).any() :
        print( 'Has Nan!', st )
        input()

    return mat

class proposal_network( network ):
    def apply( self, input, nclasses=1 ):
        nets = {}

        with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.05)) :
            nets['main'] = slim.conv2d( input, 512, [3,3],
                                        scope=self._scope_name('proposal_conv1',group='body') )
            nets['main'] = slim.conv2d( nets['main'], 512, [1,1],
                                        scope=self._scope_name('proposal_conv2',group='body') )

        with slim.arg_scope([slim.conv2d],activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.05)) :
            nets['scores'] = slim.conv2d( nets['main'], nclasses*2, [1,1],
                                            scope=self._scope_name('proposal_conv_scores',group='scores') )
            nets['rois'] = slim.conv2d( nets['main'], nclasses*4, [1,1],
                                            scope=self._scope_name('proposal_conv_rois',group='rois') )

        return {'scores' : nets['scores'], 'rois' : nets['rois'] }

    def loss( self, inputs, feat_net, proposal_net, nclasses, spatial_scale ):
        var_names = []
        var_names += self.names_group('body')
        var_names += self.names_group('scores')
        var_names += self.names_group('rois')

        labels, targets = proposal_encode.op( feat_net,
                                                inputs['gtboxes'],
                                                inputs['gtlabels'],
                                                inputs['gtbatches'],
                                                inputs['shapes'],
                                                spatial_scale=spatial_scale )


        labels, targets, mask, selection_indices, valid_indices = tf.py_func( proposal.random_selection,
                                                                [ labels, targets, inputs['batch_labels'],
                                                                nclasses,
                                                                self._dnn_cfg.PROPOSAL.TRAIN.SELECTION_BATCH_SIZE,
                                                                self._dnn_cfg.PROPOSAL.TRAIN.FG_RATIO
                                                                ],
                                                                [ tf.float32, tf.float32, tf.float32, tf.int32, tf.int32 ] )

        scores = proposal_net['scores']
        rois = proposal_net['rois']

        scores = tf.reshape( scores, [ -1, 2*nclasses ] )
        rois = tf.reshape( rois, [ -1, 4*nclasses ] )

        scores = tf.gather_nd( scores, selection_indices )
        rois = tf.gather_nd( rois, selection_indices )

        scores = tf.reshape( scores, [ -1, 2 ] )
        rois = tf.reshape( rois, [ -1, 4 ] )

        scores = tf.gather_nd( scores, valid_indices )
        rois = tf.gather_nd( rois, valid_indices )

        scores = tf.where( tf.is_nan( scores ), labels, scores )
        loss_scores = tf.nn.softmax_cross_entropy_with_logits_v2( labels=labels, logits=scores )
        loss_scores = tf.reduce_mean( loss_scores )

        rois = tf.multiply( rois, mask )
        rois = tf.where( tf.is_nan( rois ), targets, rois )
        loss_rois = tf.nn.l2_loss( rois - targets ) #/ self._dnn_cfg.PROPOSAL.TRAIN.SELECTION_BATCH_SIZE

        weights = self._dnn_cfg.PROPOSAL.TRAIN.WEIGHTS

        losses = {}
        losses['loss'] = weights['scores']*loss_scores + weights['rois']*loss_rois
        losses['scores'] = loss_scores
        losses['rois'] = loss_rois

        return losses, var_names

    def post_process( self, inputs, proposal_net, nclasses, nbatches, spatial_scale, remove_small=False ):
        scores = proposal_net['scores']
        rois = proposal_net['rois']

        all_scores = []
        all_rois = []
        all_labels = []
        all_batch_indices = []

        for i in range( nclasses ):
            # The object classes start from 1
            cls_ind = i+1
            cls_scores = tf.slice( scores, [ 0, 0, 0, i*2 ], [-1, -1, -1, 2 ] )
            cls_rois = tf.slice( rois, [ 0, 0, 0, i*4 ], [ -1, -1, -1, 4 ] )

            cls_scores = tf.where( tf.is_nan(cls_scores), tf.zeros_like(cls_scores), cls_scores )
            cls_scores = tf.nn.softmax( cls_scores )
            cls_scores = tf.where( tf.is_nan(cls_scores), tf.zeros_like(cls_scores), cls_scores )

            cls_rois = tf.where( tf.is_nan(cls_rois), tf.zeros_like(cls_rois), cls_rois )
            #cls_rois = tf.py_func( nan_check, [ cls_rois, 'before' ], tf.float32 )
            cls_rois = proposal_decode.op( cls_rois, spatial_scale=spatial_scale )
            cls_rois = tf.where( tf.is_nan(cls_rois), tf.zeros_like(cls_rois), cls_rois )
            #cls_rois = tf.py_func( nan_check, [ cls_rois, 'after' ], tf.float32 )

            cls_rois, cls_scores, cls_labels, cls_batch_inds = tf.py_func( proposal.post_processing,
                                                                [ cls_rois, cls_scores, inputs['shapes'], cls_ind,
                                                                self._dnn_cfg.PROPOSAL.SCORE_THRESH,
                                                                remove_small ],
                                                                [ tf.float32, tf.float32, tf.float32, tf.int32 ] )

            for j in range( nbatches ):
                keep = tf.py_func( common.batch_list, [ cls_batch_inds, j ], tf.int32 )

                tmp_rois = tf.gather_nd( cls_rois, keep )
                tmp_scores = tf.gather_nd( cls_scores, keep )
                tmp_labels = tf.gather_nd( cls_labels, keep )
                tmp_batch_inds = tf.gather_nd( cls_batch_inds, keep )
                scores_slice = tf.slice( tmp_scores, [0,1], [-1,1] )
                scores_slice = tf.reshape( scores_slice, [-1] )
                nms_keep = tf.image.non_max_suppression( tmp_rois,
                                                         scores_slice,
                                                         self._dnn_cfg.PROPOSAL.POST_NMS_ROIS,
                                                         self._dnn_cfg.PROPOSAL.NMS_THRESH )
                nms_keep = tf.reshape( nms_keep, [-1,1] )

                all_rois.append( tf.gather_nd( tmp_rois, nms_keep ) )
                all_scores.append( tf.gather_nd( tmp_scores, nms_keep ) )
                all_labels.append( tf.gather_nd( tmp_labels, nms_keep ) )
                all_batch_indices.append( tf.gather_nd( tmp_batch_inds, nms_keep ) )

        post_process = {}

        post_process['rois'] = tf.concat( all_rois, axis=0 )
        post_process['scores'] = tf.concat( all_scores, axis=0 )
        post_process['labels'] = tf.concat( all_labels, axis=0 )
        post_process['batch_indices'] = tf.concat( all_batch_indices, axis=0 )

        return post_process
