from easydict import EasyDict as edict
import numpy as np
from . import base as baseline

def get_cfg():
    cfg = baseline.get_cfg()

    # Modifications to baseline config
    cfg.NAME = 'conf_proposal'

    #cfg.DNN.PROPOSAL.TRAIN.SELECTION_BATCH_SIZE = {'s1':128,'s2':128}
    #cfg.DNN.FRCNN.TRAIN.SELECTION_BATCH_SIZE = {'s1':256,'s2':256,'s2_hinge':2000}

    cfg.DATASET.MIN_OBJ_SIZE = np.array( [ 15,15 ] )

    cfg.NETWORK.PROPOSAL.FEAT_NAME = {'s1':'vgg16_small','s2':'vgg16_small'}
    cfg.NETWORK.PROPOSAL.FEAT_INIT = {'s1':None,'s2':None}
    cfg.NETWORK.PROPOSAL.NITER = 300000
    cfg.NETWORK.PROPOSAL.OPTIMIZER = {}
    cfg.NETWORK.PROPOSAL.OPTIMIZER['s1'] = {'type':'GD','lr':0.01,'decay_step':100000,'decay_rate':0.1}
    cfg.NETWORK.PROPOSAL.OPTIMIZER['s2'] = {'type':'GD','lr':0.01,'decay_step':100000,'decay_rate':0.1}
    cfg.NETWORK.PROPOSAL.BATCH_SIZE = {'s1':2,'s2':2}
    cfg.NETWORK.PROPOSAL.SELECTION_BATCH_SIZE = {'s1':64,'s2':64}

    cfg.NETWORK.FRCNN.FEAT_NAME = {'s1':'vgg16_small','s2':'vgg16_small','s2_hinge':'vgg16_small'}
    cfg.NETWORK.FRCNN.FEAT_INIT = {'s1':None,'s2':None,'s2_hinge':None}
    cfg.NETWORK.FRCNN.NITER = 200000
    cfg.NETWORK.FRCNN.OPTIMIZER = {}
    cfg.NETWORK.FRCNN.OPTIMIZER['s1'] = {'type':'GD','lr':0.01,'decay_step':60000,'decay_rate':0.1}
    cfg.NETWORK.FRCNN.OPTIMIZER['s2'] = {'type':'GD','lr':0.01,'decay_step':60000,'decay_rate':0.1}
    cfg.NETWORK.FRCNN.OPTIMIZER['s2_hinge'] = {'type':'GD','lr':0.01,'decay_step':60000,'decay_rate':0.1}
    cfg.NETWORK.FRCNN.BATCH_SIZE = {'s1':2,'s2':2,'s2_hinge':2}
    cfg.NETWORK.FRCNN.SELECTION_BATCH_SIZE = {'s1':64,'s2':64,'s2_hinge':2000}

    cfg.DNN.PROPOSAL.SCORE_THRESH = 0.5
    cfg.DNN.PROPOSAL.NMS_THRESH = 0.8
    cfg.DNN.PROPOSAL.POST_NMS_ROIS = 500
    cfg.DNN.PROPOSAL.SCORE_THRESH = 0.5

    cfg.DNN.PROPOSAL.TRAIN = edict()
    cfg.DNN.PROPOSAL.TRAIN.SELECTION_BATCH_SIZE = 64
    cfg.DNN.PROPOSAL.TRAIN.FG_RATIO = 0.5
    cfg.DNN.PROPOSAL.TRAIN.REMOVE_DIFFICULTS = False
    cfg.DNN.PROPOSAL.TRAIN.WEIGHTS = {'scores':1.0, 'rois':0.01}

    cfg.DNN.FRCNN.NMS_THRESH = 0.3
    cfg.DNN.FRCNN.POST_NMS_ROIS = 500

    cfg.DNN.FRCNN.FG_MIN_OVERLAP = 0.5
    cfg.DNN.FRCNN.BG_MAX_OVERLAP = 0.3
    cfg.DNN.FRCNN.TRAIN = edict()
    cfg.DNN.FRCNN.TRAIN.SELECTION_BATCH_SIZE = 32
    cfg.DNN.FRCNN.TRAIN.FG_RATIO = 0.5
    cfg.DNN.FRCNN.TRAIN.REMOVE_DIFFICULTS = False
    cfg.DNN.FRCNN.TRAIN.PRUNE_METHOD = "random"
    cfg.DNN.FRCNN.TRAIN.ADD_GTBOXES = True
    cfg.DNN.FRCNN.TRAIN.WEIGHTS = {'scores':1.0, 'deltas':0.1}

    return cfg
