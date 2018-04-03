import numpy as np
from tools import bbox_tools

bbo = bbox_tools()

def convert_rois( rois ):
    order = np.array( [1,0,3,2] )
    rois = rois[:,order]
    return rois

def normalize_rois( rois, grid_size, base_size ) :
    rois = rois / base_size
    height = grid_size[1]
    rois = rois / ( height-1 )
    return rois

def prepare_rois( rois, grid_size, base_size ):
    rois = normalize_rois( rois, grid_size, base_size )
    return convert_rois( rois )

def prepare_cls_data( labels, targets, batch_labels, nclasses ):
    n,h,w = labels.shape[:-1]

    cls_labels = np.empty((n,h,w,nclasses), dtype=np.float32 )
    cls_labels.fill( -1.0 )
    cls_targets = np.empty((n,h,w,nclasses*4), dtype=np.float32 )
    cls_targets.fill( 0.0 )

    indices = np.arange(0,4)
    # Placing labels (b,n,m,1) into cls_labels (b,n,m,nclasses)
    for index, l in np.ndenumerate( labels ):
        i,j,k = index[:-1]

        # Setting the label
        if l >= 0 :
            cls_labels[i,j,k,:] = 0.0
            if l > 0 :
                # Setting the label
                m = int( l-1 )
                cls_labels[i,j,k,m] = 1.0

                # Copying the target
                inds = int((l-1)*4) + indices
                inds = inds.astype( int )
                cls_targets[i,j,k, inds ] = targets[i,j,k,:]

    # Removing conflicting labels
    # TODO : We need to make sure that there is a definition of compatible and
    #        none compatible objects, for now every object is regarded as
    #        non-compatible

    batch_labels = batch_labels.ravel().astype(int)
    for b, l in enumerate( batch_labels ):
        if l > 0 :
            for c in range( nclasses ) :
                if c != l-1  :
                    cls_labels[ b, :, :, c ] = -1.0

    return cls_labels, cls_targets

def random_prune( labels, targets, nclasses, batch_size, fg_ratio ):
    labels = labels.reshape([-1,nclasses])
    targets = targets.reshape([-1,nclasses*4])

    batch_size = int( batch_size )

    all_indices = []

    for i in range(nclasses) :
        cls_labels = labels[:,i]

        pos_inds = np.where( cls_labels == 1 )[0]
        neg_inds = np.where( cls_labels == 0 )[0]

        npos = int(batch_size * fg_ratio)
        pos_selection = np.array( [] )

        npos = np.min( [ len(pos_inds), npos ] )
        if npos > 0 :
            pos_selection = np.random.choice( pos_inds, npos )

        neg_selection = np.array( [] )
        nneg = batch_size - npos
        nneg = np.min( [ len(neg_inds), nneg ] )
        if nneg > 0 :
            neg_selection = np.random.choice( neg_inds, nneg )

        indices = np.concatenate( [ pos_selection, neg_selection ] ).astype( int )
        all_indices.append( indices )

    selection_indices = np.concatenate( all_indices )

    selection_indices = selection_indices.ravel()

    labels = labels[ selection_indices ]
    targets = targets[ selection_indices ]

    labels = labels.reshape( [ -1,1 ] )
    targets = targets.reshape( [-1,4 ] )

    valid_indices = np.where( labels.ravel() >= 0 )[0]
    valid_indices = valid_indices.ravel()

    return selection_indices, valid_indices

def build_category_labels( labels ):
    labels = labels.ravel()
    n = len( labels )
    clabels = np.zeros([ n,2 ] )
    for i,l in enumerate(labels) :
        clabels[i,int(l)] = 1.0
    return clabels.astype( np.float32 )

def random_selection( labels, targets, batch_labels, nclasses, batch_size, fg_ratio ):
    labels, targets = prepare_cls_data( labels, targets, batch_labels, nclasses )

    selection_indices, valid_indices = random_prune( labels, targets, nclasses, batch_size, fg_ratio )

    labels = labels.reshape([-1,nclasses])
    targets = targets.reshape([-1,nclasses*4])

    labels = labels[ selection_indices ]
    targets = targets[ selection_indices ]

    labels = labels.reshape([-1,1])
    targets = targets.reshape([-1,4])

    labels = labels[ valid_indices ]
    targets = targets[ valid_indices ]

    pos_inds = np.where( labels.ravel() == 1 )[0]

    mask = np.zeros( targets.shape ).astype( np.float32 )
    mask[ pos_inds, : ] = 1.0

    labels = build_category_labels( labels )
    selection_indices = selection_indices.reshape([-1,1]).astype( np.int32 )
    valid_indices = valid_indices.reshape([-1,1]).astype( np.int32 )

    return labels, targets, mask, selection_indices, valid_indices

def remove_small_rois( rois, min_h, min_w ):
    w = (rois[:,2] - rois[:,0]).ravel()
    h = (rois[:,3] - rois[:,1]).ravel()

    keep = np.where( ( h >= min_h ) & ( w >= min_w ) )[0];

    #if len( keep ) == 0 :
    #    keep = np.random.choice( len( rois ), 10 )

    return keep

def post_processing( rois, scores, shapes, cls_index, score_thresh, remove_small=False ):
    if np.isnan( rois ).any() :
        print('Rois has nan')

    if np.isnan( scores ).any() :
        print('Scores has nan')

    n,h,w = rois.shape[:-1]

    roi_batch_inds = np.zeros( [n,h,w,1], dtype=np.int32 )
    for i in range( n ):
        roi_batch_inds[i,:,:,:] = i

    roi_batch_inds = roi_batch_inds.reshape([-1,1])

    rois = rois.reshape([-1,4])
    scores = scores.reshape([-1,2])

    labels = np.ones( [ scores.shape[0], 1 ], dtype=np.float32 ) * ( cls_index )
    # +1 is due to the fact that the category indices start from 1
    # Making sure to at least select 10 rois in each batch for each class
    keep = []
    for i in range( n ):
        bindices = np.where( roi_batch_inds.ravel() == i )[0]
        k = np.where( scores[bindices,1].ravel() > score_thresh )[0]

        if len(k) == 0 :
            k = np.random.choice( len( bindices ), 10 )
        keep.append( bindices[k] )
    keep = np.concatenate( keep )

    roi_batch_inds = roi_batch_inds[ keep ]
    rois = rois[keep]
    scores = scores[keep]
    labels = labels[keep]

    for i in range( n ):
        inds = np.where( roi_batch_inds.ravel() == i )[0]
        rois[ inds ] = bbo.clip( rois[ inds ], shapes[i] )

        #print( i, np.max( rois[inds], axis=0 ) )

    if remove_small :
        keep = remove_small_rois( rois, 15, 15 )
        rois = rois[keep]
        scores = scores[keep]
        labels = labels[keep]
        roi_batch_inds = roi_batch_inds[keep]

    return rois, scores, labels, roi_batch_inds
