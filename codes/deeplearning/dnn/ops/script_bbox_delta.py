import pickle
import sys
import numpy as np
import bbox_delta
import tensorflow as tf

def add_path( p ):
    if p not in sys.path :
        sys.path.append( p )

add_path('../common')
add_path('../cpp_common/build')

from tools import bbox_tools

bbo = bbox_tools()

if __name__=="__main__" :
    with open('blobs.pkl','rb') as ff :
        blobs = pickle.load( ff )[0]

    labels = blobs['roi_labels']
    rois = blobs['rois']
    deltas = blobs['roi_deltas']
    targets = blobs['roi_targets']

    labels = np.argmax( labels, axis=1 )
    pos_I = np.where( labels == 1 )[0]

    dd = bbo.transform( rois[pos_I,1:], targets[pos_I] )
    tt = bbo.transform_inv( rois[pos_I,1:], dd )
    print(np.linalg.norm(tt - targets[pos_I,:]))


    #indices = rois[:,0].reshape([-1,1])
    #targets = np.concatenate([ indices, targets ], axis=1 )

    #out = bbo.transform( rois[:,1:], targets )

    #print( out )


    rois_tensor = tf.placeholder( tf.float32, [ None, 5 ] )
    detlas_tensor = tf.placeholder( tf.float32, [ None, 4 ] )

    out_tensor = bbox_delta.inv_op( rois_tensor, detlas_tensor )

    sess = tf.Session()
    out = sess.run( out_tensor, feed_dict={ rois_tensor:rois[ pos_I,: ], detlas_tensor:deltas[ pos_I,: ] } )
    
    print( out[:,1:] )
    print( targets[pos_I,:] )