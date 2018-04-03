import _init
import numpy as np
import argparse
import time
import pickle
import tensorflow as tf
import tensorflow.contrib.slim as slim
from pprint import pprint
from pose_config import config

from data import workbench
from dnn.blobs import blobs_generator
from tools import batch_generator
from dnn.interfaces import proposal_interface
import learning

class trainer( learning.trainer ) :
    def train_end2end( self ):
        mcfg = self._cfg.mcfg

        self._blobs_gen = blobs_generator[ self._model_name ]( self._cfg )
        self._batch_gen = batch_generator( self._bench.dataset,
                                            mcfg['batch_size'],
                                            mcfg['use_negatives'],
                                            nclasses=mcfg['nclasses'],
                                            min_size=self._cfg.dataset.MIN_OBJ_SIZE )

        proposal = proposal_interface( mcfg, self._cfg.dnn, nlevels=5 )
        losses, inputs, variables = proposal.train_loss_end2end()

        optimizer, global_step = self._get_optimizer( mcfg['optimizer'] )
        train_step = optimizer.minimize( losses['loss'], var_list=variables, global_step=global_step )

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        #if mcfg['pre'] is not None :
        #    print('Loading ', mcfg['pre'] )
        #    detector.load( sess, mcfg['pre'] )

        self._train( train_step, sess, mcfg, losses, inputs )

        detector.save( sess, mcfg['path'] )
        print("Saved to ", mcfg['path'])

def get_data():
    params = {'cfg_name': 'conf_proposal',
              'model_name': 'end2end_s1',
              'dset_params': ['coco', 'person', 'train2014'],
              'nclasses': 1,
              'batch_size': 4,
              'niter': None, 'tag':
              '20171209',
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

    cfg = config( params['cfg_name'] )
    bench = workbench( cfg, params, mcfg_mod )
    bench.load_dataset()

    mcfg = cfg.mcfg

    blobs_gen = blobs_generator[ params['model_name'] ]( mcfg, cfg.dnn )
    batch_gen = batch_generator( bench.dataset,
                                 mcfg['batch_size'],
                                 True,
                                 nclasses=mcfg['nclasses'] )

    return batch_gen, blobs_gen

if __name__=="__main__" :
    params = _init.parse_commandline()

    mcfg_mod = {}
    mcfg_mod['batch_size'] = int( params['batch_size'] )
    mcfg_mod['nclasses'] = int( params['nclasses'] )

    if not params['tag'] is None :
        mcfg_mod['tag'] = params['tag']

    if not params['prefix'] is None :
        mcfg_mod['prefix'] = params['prefix']

    if params['niter'] is not None :
        mcfg_mod['niter'] = int( params['niter'] )

    cfg = config( params['cfg_name'] )
    bench = workbench( cfg, params, mcfg_mod )
    bench.load_dataset()

    tt = trainer( cfg, bench, params['model_name'] )
    tt.train_end2end()
