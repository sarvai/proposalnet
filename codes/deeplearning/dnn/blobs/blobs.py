import numpy as np
#from config import dnn_cfg
from tools import image_tools
from PIL import Image

class data_blobs :
    def __init__( self, dnn_cfg ):
        self._dnn_cfg = dnn_cfg

    def _remove_small_boxes( self, boxes, min_size ):
        if len( boxes ) == 0 :
            return boxes

        h = boxes[:,3] - boxes[:,1]
        w = boxes[:,2] - boxes[:,0]

        inds = np.where( ( w >= min_size ) & ( h >= min_size ) )
        return boxes[ inds ]

    def _add_data( self, images, blobs, max_size=None ):

        data = []
        shapes = []
        scales = []
        batch_labels = []

        if max_size is None :
            max_size = self._dnn_cfg.TRAIN.MAX_SIZE

        for image in images :
            # Placing the scaled image in the frame
            #imsize = image.shape
            #scale = image_tools.image_scale( imsize, max_size )

            # Fitting the image to the frame
            image.scale = 1.0
            imsize = image.shape
            scale = image_tools.image_scale_fit( imsize, max_size )

            image.scale *= scale
            im = image.im

            d = np.zeros( [ max_size, max_size, 3 ] ).astype( np.float32 )
            h,w = im.shape[:2]

            d[:h,:w,:] = im

            data.append( d )
            shapes.append( image.shape )
            scales.append( image.scale )
            batch_labels.append( image.label )

        blobs['data'] = np.array( data, dtype=np.float32 )
        blobs['shapes'] = np.array( shapes, dtype=np.float32 )
        blobs['scales'] = np.array( scales, dtype=np.float32 )
        blobs['batch_labels'] = np.array( batch_labels, dtype=np.float32 ).reshape( [-1,1] )

    def _add_data_deploy( self, image, blobs ):
        data = []
        shapes = []
        scales = []

        max_size = self._dnn_cfg.DEPLOY.MAX_SIZE
        scale = image_tools.image_scale_fit( image.shape, max_size )

        image.scale *= scale
        im = image.im

        d = np.zeros( [ max_size, max_size, 3 ] ).astype( np.float32 )
        h,w = im.shape[:2]

        d[:h,:w,:] = im

        data.append( d )
        shapes.append( image.shape )
        scales.append( image.scale )

        blobs['data'] = np.array( data, dtype=np.float32 )
        blobs['shapes'] = np.array( shapes, dtype=np.float32 )
        blobs['scales'] = np.array( scales, dtype=np.float32 )

    def _add_data_pil( self, images, blobs, max_size ):
        data = []
        shapes = []
        scales = []

        for image in images :
            scale = image_tools.image_scale_fit( image.size[::-1], max_size )

            w = int(np.floor( image.size[0]*scale ))
            h = int(np.floor( image.size[1]*scale ))
            image = image.resize( [w,h], Image.BILINEAR )

            image = np.array( image )
            d = np.zeros( [ max_size, max_size, 3 ] ).astype( np.float32 )
            h,w = image.shape[:2]
            d[:h,:w,:] = image

            data.append( d )
            shapes.append( image.shape[:2] )
            scales.append( scale )

        blobs['data'] = np.array( data, dtype=np.float32 )
        blobs['shapes'] = np.array( shapes, dtype=np.float32 )
        blobs['scales'] = np.array( scales, dtype=np.float32 )

    def _add_data_deploy_ndarray( self, image, blobs ):
        data = []
        shapes = []
        scales = []

        max_size = self._dnn_cfg.DEPLOY.MAX_SIZE
        scale = image_tools.image_scale_fit( image.shape, max_size )

        img = Image.fromarray( image )
        w = int(np.floor( img.size[0]*scale ))
        h = int(np.floor( img.size[1]*scale ))
        img = img.resize( [w,h], Image.BILINEAR )

        image = np.array( img )

        d = np.zeros( [ max_size, max_size, 3 ] ).astype( np.float32 )
        h,w = image.shape[:2]
        d[:h,:w,:] = image

        data.append( d )
        shapes.append( image.shape[:2] )
        scales.append( scale )

        blobs['data'] = np.array( data, dtype=np.float32 )
        blobs['shapes'] = np.array( shapes, dtype=np.float32 )
        blobs['scales'] = np.array( scales, dtype=np.float32 )

    def _add_gtboxes( self, images, blobs, remove_difficults=False ):
        #min_obj_side = self._dnn_cfg.TRAIN.MIN_OBJ_SIDE

        gtboxes = []
        gtbatches = []
        gtlabels = []

        for i,image in enumerate( images ) :
            boxes = image.gtboxes
            # Removing Small Boxes
            #boxes = self._remove_small_boxes( boxes, min_obj_side )
            labels = np.ones((len(boxes),1)) * image.label
            #difficult = image.difficult

            if len( boxes ) > 0 :

                #if remove_difficults :
                #    inds = np.where( difficult == 1.0 )[0]
                #    labels[ inds ] = -1

                indices = np.ones((len(boxes),1)) * i

                gtboxes.append( boxes )
                gtbatches.append( indices )
                gtlabels.append( labels )


        if len( gtboxes ) > 0 :
            gtboxes = np.concatenate( gtboxes )
            gtbatches = np.concatenate( gtbatches )
            gtlabels = np.concatenate( gtlabels )
        else :
            gtboxes = np.zeros([0,4])
            gtbatches = np.zeros([0,1])
            gtlabels = np.zeros([0,1])

        blobs['gtboxes'] = gtboxes
        blobs['gtbatches'] = gtbatches.astype( np.int32 )
        blobs['gtlabels'] = gtlabels

    def get_blobs( self, images ):
        blobs = {}

        self._add_data( images, blobs )
        self._add_gtboxes( images, blobs )

        return blobs

    def get_blobs_deploy_pil( self, images, max_size=None ):
        if max_size is None :
            max_size = self._dnn_cfg.DEPLOY.MAX_SIZE
        blobs = {}
        self._add_data_pil( images, blobs, max_size )
        return blobs
