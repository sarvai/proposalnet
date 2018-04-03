import numpy as np
from tools import bbox_tools

bbo = bbox_tools()

def batch_list( batch_indices, batch_id ):
    batch_indices = batch_indices.ravel()
    inds = np.where( batch_indices == batch_id )[0]
    return inds.reshape([-1,1]).astype(np.int32)

def convert_rois( rois ):
    order = np.array( [1,0,3,2] )
    rois = rois[:,order]
    return rois

def class_indices( labels, cind ):
    labels = labels.ravel()
    indices = np.where( labels == cind )[0]
    indices = np.array( indices, dtype=np.int32 ).reshape([-1,1])
    return indices

def clip_rois( rois, batch_indices, shapes ):
    n = len(shapes)
    for i in range( n ):
        inds = np.where( batch_indices.ravel() == i )[0]
        rois[ inds ] = bbo.clip( rois[ inds ], shapes[i] )
    return rois

def normalize_rois( rois, grid_size, spatial_scale ) :
    rois = rois * spatial_scale
    height = grid_size[1]
    rois = rois / ( height-1 )
    return rois

def prepare_rois( rois, grid_size, spatial_scale ):
    rois = normalize_rois( rois, grid_size, spatial_scale )
    return convert_rois( rois )
