import numpy as np
import tensorflow as tf
from pprint import pprint
import time

class trainer :
    def __init__( self, cfg, bench, model_name ):
        self._cfg = cfg
        self._bench = bench
        self._model_name = model_name

        self._batch_gen = None
        self._blobs_gen = None

    def _get_next_blobs( self ):
        done = False

        while not done :
            images = self._batch_gen.next()
            blobs = self._blobs_gen.get_blobs( images )

            if 'valid' in blobs :
                if blobs['valid'] :
                    done = True
            else :
                done = True

        return blobs

    def _get_optimizer( self, params ):
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay( params['lr'],
                                                    global_step, params['decay_step'],
                                                    params['decay_rate'],
                                                    staircase=params.get('staircase',True) )

        if params['type'] == 'Adam' :
            optimizer = tf.train.AdamOptimizer( learning_rate )
        elif params['type'] == 'GD':
            optimizer = tf.train.GradientDescentOptimizer( learning_rate )
        elif params['type'] == 'RMS':
            optimizer = tf.train.RMSPropOptimizer( learning_rate )
        else :
            raise NotImplemented

        return optimizer, global_step

    def _blobs2feed_dict( self, inputs, blobs ):
        feed_dict = {}
        for k in inputs.keys() :
            feed_dict[ inputs[k] ] = blobs[k]
        return feed_dict

    def _train( self, train_step,  sess, losses, inputs, snapshots=None ):
        t0 = time.time()
        for i in range( self._cfg.dnn.TRAIN.NITER ):    
            if (i+1) % 100 == 0 :
                t1 = time.time()

                blobs = self._get_next_blobs()
                if not 'dropout_prob' in blobs.keys() :
                    blobs['dropout_prob'] = 1.0
                blobs['iteration'] = np.int32(i)
                l = sess.run( losses, feed_dict=self._blobs2feed_dict(inputs,blobs) )
                print("Interation : %d, Iters/sec : %g, loss : %g" % (i+1, 100.0/(t1-t0), l['loss'] ) )
                pprint( l )
                t0 = time.time()

            blobs = self._get_next_blobs()
            if not 'dropout_prob' in blobs.keys() :
                blobs['dropout_prob'] = 0.5
            blobs['iteration'] = np.int32(i)
            feed_dict = self._blobs2feed_dict(inputs,blobs)
            train_step.run( feed_dict=feed_dict, session=sess )

            if snapshots is not None :
                ii = i+1

                if ii < snapshots['step']['thresh'] :
                    step = snapshots['step']['below']
                else :
                    step = snapshots['step']['above']

                if ii % step == 0 :
                    path = snapshots['tmp'] % ( ii )
                    print('Saving snapshot to %s' % (path) )
                    snapshots['interface'].save( sess, path )
