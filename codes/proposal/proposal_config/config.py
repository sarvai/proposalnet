from easydict import EasyDict as edict
import sys
import os
import platform
import numpy as np
from pprint import pprint
from .models_cfg import models_cfg

def mkdir( d ):
    if not os.path.isdir( d ) :
        os.mkdir( d )

class config :
    def __init__( self, cfg_name ):
        self._cfg_name = cfg_name
        self._cfg = None
        self._mcfg = None

        self.initialize()
        self.initialize_dataset()
        self.initialize_data()

    def initialize( self ):
        from .configs import configurations
        assert self._cfg_name in configurations, "Unknown config"

        cfg = configurations[ self._cfg_name ].get_cfg()
        assert cfg.NAME == self._cfg_name, "Config name does not match"

        self._cfg = cfg

    def initialize_dataset( self ):
        if platform.system() == 'Darwin' :
            self._cfg.DATASET.ROOT = "/Users/heydar/Work/void/data"
        else :
            self._cfg.DATASET.ROOT = "/ssd/data"

        self._cfg.DATASET.ANNOTS_TMP = "%s/annotations/%s.pkl" % ( self._cfg.DATASET.ROOT, '%s' )
        self._cfg.DATASET.IMAGES_TMP = "%s/datasets/%s" % ( self._cfg.DATASET.ROOT,'%s' )
        self._cfg.DATASET.WEIGHTS_TMP = "%s/weights/%s" % ( self._cfg.DATASET.ROOT, '%s' )

    def initialize_data( self ):
        tmp = edict()

        tmp.PROJECT = "proposal"
        tmp.TAG = self._cfg.NAME

        tmp.DIRS = edict()
        tmp.DIRS.BASE = ""
        if platform.system() == "Linux" :
            tmp.DIRS.BASE = os.path.join( "/network/data", tmp.PROJECT )
        else :
            tmp.DIRS.BASE = os.path.join( "/Users/heydar/Work/void", tmp.PROJECT )

        mkdir( tmp.DIRS.BASE )

        tmp.DIRS.ROOT = os.path.join( tmp.DIRS.BASE, tmp.TAG )
        mkdir( tmp.DIRS.ROOT )
        tmp.DIRS.MODELS = os.path.join( tmp.DIRS.ROOT, "models" )
        mkdir( tmp.DIRS.MODELS )
        tmp.DIRS.DETS = os.path.join( tmp.DIRS.ROOT, "dets" )
        mkdir( tmp.DIRS.DETS )
        tmp.DIRS.SNAPSHOTS = os.path.join( tmp.DIRS.ROOT, "snapshots" )
        mkdir( tmp.DIRS.SNAPSHOTS )
        tmp.DIRS.BENCHMARK = os.path.join( tmp.DIRS.ROOT, "benchmark" )
        mkdir( tmp.DIRS.BENCHMARK )

        self._cfg.DATA = tmp

    def build_models_cfg( self, model_name, tag, data_hash, model_hash, mod=False ) :
        self._mcfg = models_cfg[ model_name ]( self._cfg, tag, data_hash, model_hash, mod )

    @property
    def dnn( self ):
        return self._cfg.DNN

    @property
    def data( self ):
        return self._cfg.DATA

    @property
    def dataset( self ):
        return self._cfg.DATASET

    @property
    def mcfg( self ):
        return self._mcfg
