import numpy as np
import copy
from pprint import pprint

from .image import image
from tools import annot_tools

class deepfashion_image( image ):
    def __init__( self, cfg, image_info, mirrored=False ):
        super().__init__( cfg, image_info, mirrored )

        self.label = image_info.get('label')

        self._data['gtboxes'] = []
        self._data['category'] = []

        for obj in self._image_info['objects'] :
            self._data['gtboxes'].append( obj.get('bbox') )
            self._data['category'].append( obj.get('category') )

        self._data['gtboxes'] = np.array( self._data['gtboxes'], dtype=np.float32 )
        self._data['category'] = np.array( self._data['category'], dtype=np.float32 )
        self._data['category'] = self._data['category'].reshape([-1,1])

    @property
    def category( self ):
        if 'category' in self._data :
            return copy.deepcopy( self._data['category'] )
        else :
            return np.array([])

    def show_info( self ):
        pprint( self._image_info )
