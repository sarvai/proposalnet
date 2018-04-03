import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pickle

import frcnn_prepare
import overlaps

def whc( b ):
    w = b[2] - b[0]
    h = b[3] - b[1]
    x = b[0] + w*0.5
    y = b[1] + h*0.5
    return np.array( [w,h,x,y] )

def intersection_area( b0, b1 ):
    ix = np.min( [ b0[2], b1[2] ] ) - np.max( [ b0[0], b1[0] ] )
    iy = np.min( [b0[3], b1[3] ] ) - np.max( [ b0[1], b1[1] ] )
    intersection = np.max( [ 0, ix+1 ] ) * np.max( [ 0, iy+1 ] )
    return intersection

def area( b ):
    return ( b[2] - b[0] ) * ( b[3] - b[1] )

def iou( r0, r1 ):
    a0 = area(r0)
    a1 = area(r1)
    ii = intersection_area( r0, r1 )
    return ii / ( a0+a1-ii )

def py_overlaps( rois, boxes ):
    ovs = np.zeros((len(rois), len(boxes)))

    for i, roi in enumerate( rois ):
        for j, box in enumerate( boxes ):
            ovs[i,j] = iou( roi, box )

    return ovs

def forward_transform( r,t ):
    r_whc = whc( r )
    t_whc = whc( t )

    dx = ( t_whc[2] - r_whc[2] ) / r_whc[0]
    dy = ( t_whc[3] - r_whc[3] ) / r_whc[1]
    dw = np.log(t_whc[0] / r_whc[0])
    dh = np.log(t_whc[1] / r_whc[1])

    return np.array([ dx,dy,dw,dh ])

def transform( refs, targets ):
    #
    # Refs are the boxes that already exist
    # Targets are the boxes that we wish to predict
    #

    assert refs.shape == targets.shape, "The shape of the references and the targets do not match"

    #for r,t in zip( refs, targets ):
    #    d = forward( r,t )
    #    t2 = backward( r,d )
    #    print( t,t2 )

    delta = []

    for r,t in zip( refs, targets ):
        delta.append( forward_transform(r,t) )

    return np.array( delta, dtype=np.float32 )

class frcnn_prepare_test :
    def __init__( self ):
        with open('frcnn_blobs.pkl','rb') as ff :
            self._blobs = pickle.load(ff)[0]
        self._blobs['gtbatches'] = self._blobs['gtbatches'].astype( np.int32 )
        self._blobs['roibatches'] = self._blobs['roibatches'].astype( np.int32 )
        self._blobs['gtlabels'] = self._blobs['gtlabels'] - 1

        for name, item in self._blobs.items() :
            print( name, item.shape )

        self._nclasses = 2
        self._nbatches = 4
        self._fg_min_overlap = 0.5
        self._bg_max_overlap = 0.2

    def run_python( self ):

        count = 0

        nrois = len( self._blobs['rois'] )
        labels = np.empty((nrois,self._nclasses))
        labels.fill(-1.0)

        gtboxes = self._blobs['gtboxes']
        gtlabels = self._blobs['gtlabels']
        rois = self._blobs['rois']
        roilabels = self._blobs['roilabels']

        clsrois = np.zeros((nrois,self._nclasses*4))
        targets = np.zeros((nrois,self._nclasses*4))
        deltas = np.zeros((nrois,self._nclasses*4))

        for batch_ind in range( self._nbatches ):
            gtinds = np.where( self._blobs['gtbatches'].ravel() == batch_ind )[0]
            roiinds = np.where( self._blobs['roibatches'].ravel() == batch_ind )[0]

            batch_gtboxes = self._blobs['gtboxes'][gtinds]
            batch_gtlabels = self._blobs['gtlabels'][gtinds].ravel()
            batch_rois = self._blobs['rois'][roiinds]
            batch_roilabels = self._blobs['roilabels'][roiinds].ravel()

            for i in range( self._nclasses ):
                gtclsinds = np.where( batch_gtlabels == i )[0]
                roiclsinds = np.where( batch_roilabels == i )[0]

                if len(gtclsinds)>0 and len(roiclsinds)>0 :
                    ovs = py_overlaps( batch_rois[roiclsinds], batch_gtboxes[gtclsinds] )
                    maxes = np.max( ovs, axis=1 )
                    argmaxes = np.argmax( ovs, axis=1 )

                    n_pos = np.where( maxes >= self._fg_min_overlap )[0]
                    n_neg = np.where( maxes < self._bg_max_overlap )[0]

                    inds = roiinds[ roiclsinds[ n_pos ] ]
                    labels[ inds, i ] = 1.0
                    p0 = 4*i
                    p1 = 4*(i+1)


                    clsrois[ inds, p0:p1 ] = rois[ inds ]
                    targets[ inds, p0:p1 ] = batch_gtboxes[ argmaxes[ n_pos ] ]
                    deltas[ inds, p0:p1 ] = transform( rois[ inds ], batch_gtboxes[ argmaxes[ n_pos ] ] )

                    inds = roiinds[ roiclsinds[ n_neg ] ]
                    labels[ inds, i ] = 0.0


                    n = len(n_pos) + len(n_neg)

                    count = count + n

        maxes = np.max( labels, axis=1 )
        inds = np.where( maxes >= 0 )[0]

        return labels, deltas

    def run_tf( self ):
        inputs = {}
        inputs['gtboxes'] = tf.placeholder( tf.float32, shape=[ None, 4 ] )
        inputs['gtlabels'] = tf.placeholder( tf.float32, shape=[ None, 1 ] )
        inputs['gtbatches'] = tf.placeholder( tf.int32, shape=[ None, 1 ] )

        inputs['rois'] = tf.placeholder( tf.float32, shape=[ None, 4 ] )
        inputs['roilabels'] = tf.placeholder( tf.float32, shape=[ None, 1 ] )
        inputs['roibatches'] = tf.placeholder( tf.int32, shape=[ None, 1 ] )

        ovs = overlaps.op( inputs['gtboxes'], inputs['gtlabels'], inputs['gtbatches'],
                                                     inputs['rois'], inputs['roilabels'], inputs['roibatches'] )

        labels, deltas = frcnn_prepare.op( inputs['gtboxes'], inputs['gtlabels'], inputs['gtbatches'],
                                                     inputs['rois'], inputs['roilabels'], inputs['roibatches'],
                                                     ovs, nclasses=self._nclasses, fg_min_overlap=self._fg_min_overlap,
                                                     bg_max_overlap=self._bg_max_overlap )

        outputs = {}
        outputs['labels'] = labels
        outputs['deltas'] = deltas

        sess = tf.Session()

        feed_dict = {}
        for name, var in inputs.items() :
            feed_dict[var] = self._blobs[name]

        oblobs = sess.run( outputs, feed_dict=feed_dict )

        return oblobs['labels'], oblobs['deltas']

    def do_stuff( self ):
        py_labels, py_deltas = self.run_python()
        tf_labels, tf_deltas = self.run_tf()

        print( "labels difference ", np.linalg.norm( py_labels - tf_labels ) )
        print( "deltas difference ", np.linalg.norm( py_deltas - tf_deltas ) )

if __name__=="__main__" :
    fpt = frcnn_prepare_test()
    fpt.do_stuff()
