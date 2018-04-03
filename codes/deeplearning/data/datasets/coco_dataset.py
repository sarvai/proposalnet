import numpy as np
from .dataset import dataset
from ..image import coco_image

class coco_dataset( dataset ):
    def __init__( self, cfg, cls, part, img_type=coco_image ):
        super().__init__( cfg, img_type )

        self._dset_tag = 'coco_%s_%s' % ( cls, part )

        c = cls[:2]
        if part == 'train2014' :
            p = 'tr14'
        elif part == 'val2014' :
            p = 'va14'
        else :
            p = 'te'

        self._dset_hash = 'C%s%s' % ( c, p )
        self._data_name = 'coco_%s' % ( cls )
        self._images = []

    def load( self, setting=None ):
        super().load( setting=setting )

    def reduce_to_pose( self ):
        keep = []
        for ind, image in enumerate(self._original_images) :
            labels, points = image.keypoints
            labels = labels.ravel()
            valid = np.where( labels > 0 )[0]

            if len( valid ) > 0 :
                keep.append( ind )

        self._original_images = self._original_images[keep]
