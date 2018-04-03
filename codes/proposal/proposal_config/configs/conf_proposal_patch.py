from easydict import EasyDict as edict
from . import base as baseline

def get_cfg():
    cfg = baseline.get_cfg()

    # Modifications to baseline config
    cfg.NAME = 'conf_proposal_patch'
    cfg.DATASET.WORKSET_SETTING = ['add_mirrored']
    #cfg.DNN.PROPOSAL.TRAIN.SELECTION_BATCH_SIZE = {'s1':128,'s2':128}
    #cfg.DNN.FRCNN.TRAIN.SELECTION_BATCH_SIZE = {'s1':256,'s2':256,'s2_hinge':2000}

    cfg.NETWORK.PROPOSAL.FEAT_NAME = {'s1':'vgg16_very_small','s2':'vgg16_very_small'}
    cfg.NETWORK.PROPOSAL.FEAT_INIT = {'s1':None,'s2':None}
    cfg.NETWORK.PROPOSAL.NITER = 300000
    cfg.NETWORK.PROPOSAL.OPTIMIZER = {}
    cfg.NETWORK.PROPOSAL.OPTIMIZER['s1'] = {'type':'GD','lr':0.01,'decay_step':100000,'decay_rate':0.1}
    cfg.NETWORK.PROPOSAL.OPTIMIZER['s2'] = {'type':'GD','lr':0.01,'decay_step':100000,'decay_rate':0.1}
    cfg.NETWORK.PROPOSAL.BATCH_SIZE = {'s1':2,'s2':2}
    cfg.NETWORK.PROPOSAL.SELECTION_BATCH_SIZE = {'s1':64,'s2':64}

    cfg.DNN.TRAIN.MAX_SIZE = 64

    cfg.DNN.PROPOSAL.SCORE_THRESH = 0.5
    cfg.DNN.PROPOSAL.NMS_THRESH = 0.8
    cfg.DNN.PROPOSAL.POST_NMS_ROIS = 500
    cfg.DNN.PROPOSAL.SCORE_THRESH = 0.5

    cfg.DNN.PROPOSAL.TRAIN = edict()
    cfg.DNN.PROPOSAL.TRAIN.SELECTION_BATCH_SIZE = 32
    cfg.DNN.PROPOSAL.TRAIN.FG_RATIO = 0.5
    cfg.DNN.PROPOSAL.TRAIN.REMOVE_DIFFICULTS = False
    cfg.DNN.PROPOSAL.TRAIN.WEIGHTS = {'scores':1.0, 'rois':0.01}

    cfg.DNN.FRCNN.NMS_THRESH = 0.3
    cfg.DNN.FRCNN.POST_NMS_ROIS = 500

    cfg.DNN.FRCNN.FG_MIN_OVERLAP = 0.5
    cfg.DNN.FRCNN.BG_MAX_OVERLAP = 0.3

    cfg.DNN.FRCNN.POOL_HEIGHT = 2
    cfg.DNN.FRCNN.POOL_WIDTH = 2
    cfg.DNN.FRCNN.FC_SIZE = 512

    cfg.DNN.FRCNN.TRAIN = edict()
    cfg.DNN.FRCNN.TRAIN.SELECTION_BATCH_SIZE = 16
    cfg.DNN.FRCNN.TRAIN.FG_RATIO = 0.5
    cfg.DNN.FRCNN.TRAIN.REMOVE_DIFFICULTS = False
    cfg.DNN.FRCNN.TRAIN.PRUNE_METHOD = "random"
    cfg.DNN.FRCNN.TRAIN.ADD_GTBOXES = True
    cfg.DNN.FRCNN.TRAIN.WEIGHTS = {'scores':1.0, 'deltas':0.1}

    return cfg
