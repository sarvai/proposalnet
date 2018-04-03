from .dataset import dataset
from ..image import coco_image

class cocowider_dataset( dataset ):
    def __init__( self, cfg, cls, part, img_type=coco_image ):
        super().__init__( cfg, img_type )

        self._dset_tag = 'cocowider_%s_%s' % ( cls, part )

        c = cls[:2]
        if part == 'train' :
            p = 'tr'
        elif part == 'val' :
            p = 'va'
        else :
            p = 'te'

        self._dset_hash = 'CW%s%s' % ( c, p )
        self._data_name = 'cocowider_%s' % ( cls )
        self._images = []

    def load( self, setting=None ):
        super().load( setting=setting )
