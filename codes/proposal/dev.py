import _init
import numpy as np
import argparse
import time
import pickle
import tensorflow as tf
import tensorflow.contrib.slim as slim
from pprint import pprint

from data import workbench
from proposal_config import config
from dnn.blobs import blobs_generator
from tools import batch_generator
from dnn.interfaces import proposal_interface

import learning

from tools import pil_tools
from PIL.ImageDraw import Draw

class dev :
    def __init__( self ):
        params = {'cfg_name': 'conf_proposal',
              'model_name': 'end2end_s1', 
              'dset_params': ['coco', 'person', 'train2014'], 
              'nclasses': 1, 
              'batch_size': 4, 
              'niter': None, 'tag': 
              '20171211', 
              'prefix': 'person'}

        mcfg_mod = {}
        mcfg_mod['batch_size'] = int( params['batch_size'] )
        mcfg_mod['nclasses'] = int( params['nclasses'] )

        if not params['tag'] is None :
            mcfg_mod['tag'] = params['tag']

        if not params['prefix'] is None :
            mcfg_mod['prefix'] = params['prefix']

        if params['niter'] is not None :
            mcfg_mod['niter'] = int( params['niter'] )

        self._cfg = config( params['cfg_name'] )
        self._bench = workbench( self._cfg, params, mcfg_mod )
        self._bench.load_dataset()
 
        self._blobs_gen = blobs_generator[ params['model_name'] ]( self._cfg )
        self._batch_gen = batch_generator( self._bench.dataset, 
                                 self._cfg.mcfg['batch_size'], 
                                 True, 
                                 nclasses=self._cfg.mcfg['nclasses'],
                                 min_size=self._cfg.dataset.MIN_OBJ_SIZE )

    def load_model( self ):
        tf.reset_default_graph()

        self._detector = proposal_interface( self._cfg.mcfg, self._cfg.dnn )
        self._model = self._detector.deploy()

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        self._detector.load( self._sess, self._cfg.mcfg['path'] )

    def fix_batch( self ):
        self._batch = self._batch_gen.next()

    def do_stuff( self ):
        blobs = self._blobs_gen.get_blobs( self._batch )

        print( blobs['scales'] )
        out = self._detector.eval( self._sess, blobs, self._model )

        res = {}
        res['images'] = self._batch
        res['output'] = out
        res['scales'] = blobs['scales']

        return res

    def visualize( self, res, batch_index, thresh=0.95 ):
        #print( res.keys() )
        #print( batch_index )

        image = res['images'][ batch_index ]
        image.scale = 1.0

        batch_indices = res['output']['batch_indices'].ravel()
        inds = np.where( batch_indices == batch_index )[0]

        scores = res['output']['scores'][ inds ][:,1].ravel()
        rois = res['output']['rois'][ inds ] / res['scales'][ batch_index ]

        keep = np.where( scores >= 0.95 )[0]

        scores = scores[ keep ]
        rois = rois[ keep ]

        im = image.im_PIL
        draw = Draw(im)

        for r in rois :
            pil_tools.draw_rectangle( draw, r )

        return np.array(im)
