from easydict import EasyDict as edict
import numpy as np

def get_cfg() :
    cfg = edict()

    cfg.NAME = 'baseline'

    cfg.DATA = None

    cfg.DATASET = edict()
    cfg.DATASET.MIN_OBJ_SIZE = np.array( [ 40,40 ] )
    cfg.DATASET.SCALE = False
    cfg.DATASET.SCALES = [ 0.75, 2.0 ]
    cfg.DATASET.WORKSET_SETTING = ['add_mirrored','randomize_scales','prune_by_size']
    cfg.DATASET.WORKSET_SETTING_DEPLOY = ['add_mirrored']
    
    cfg.DNN = edict()

    cfg.DNN.COMMON = edict()
    cfg.DNN.COMMON.PIXEL_MEANS = np.array( [[[102.9801, 115.9465, 122.7717]]], dtype=np.float32 )
    cfg.DNN.COMMON.COLORSPACE = 'rgb'
    cfg.DNN.COMMON.FEAT_MAP_SCALE = -1#1.0/16.0
    cfg.DNN.COMMON.BASE_SIZE = -1#16.0

    cfg.DNN.ANCHORS = edict()
    cfg.DNN.ANCHORS.RATIOS = [ 0.25, 0.5, 1, 2, 4 ]
    cfg.DNN.ANCHORS.SCALES = [ 4, 8, 16 ]
    cfg.DNN.ANCHORS.GT_OVERLAP = 0.2
    cfg.DNN.ANCHORS.NUM_ANCHORS = len( cfg.DNN.ANCHORS.RATIOS ) * len( cfg.DNN.ANCHORS.SCALES )
    cfg.DNN.ANCHORS.NMS_THRESH = 0.8

    cfg.DNN.TRAIN = edict()
    cfg.DNN.TRAIN.MAX_SIZE = 800
    cfg.DNN.TRAIN.BATCH_SIZE = 2
    cfg.DNN.TRAIN.RANDOM_SCALE = True
    cfg.DNN.TRAIN.SCALE_RANGE = [ 0.5, 1.5 ]
    cfg.DNN.TRAIN.MIN_OBJ_SIDE = 10

    cfg.DNN.DEPLOY = edict()
    cfg.DNN.DEPLOY.MAX_SIZE = 800

    cfg.DNN.PROPOSAL = edict()

    cfg.DNN.PROPOSAL.SCORE_THRESH = 0.5
    cfg.DNN.PROPOSAL.NMS_THRESH = 0.8
    cfg.DNN.PROPOSAL.POST_NMS_ROIS = 500
    cfg.DNN.PROPOSAL.SCORE_THRESH = 0.5

    cfg.DNN.PROPOSAL.TRAIN = edict()
    cfg.DNN.PROPOSAL.TRAIN.SELECTION_BATCH_SIZE = 64
    cfg.DNN.PROPOSAL.TRAIN.FG_RATIO = 0.5
    cfg.DNN.PROPOSAL.TRAIN.REMOVE_DIFFICULTS = False
    cfg.DNN.PROPOSAL.TRAIN.WEIGHTS = {'scores':1.0, 'rois':0.01}

    cfg.DNN.FRCNN = edict()

    cfg.DNN.FRCNN.NMS = True
    cfg.DNN.FRCNN.NMS_THRESH = 0.8
    cfg.DNN.FRCNN.POST_NMS_ROIS = 500

    cfg.DNN.FRCNN.FG_MIN_OVERLAP = 0.5
    cfg.DNN.FRCNN.BG_MAX_OVERLAP = 0.2
    cfg.DNN.FRCNN.POOL_HEIGHT = 6
    cfg.DNN.FRCNN.POOL_WIDTH = 6
    cfg.DNN.FRCNN.FC_SIZE = 4096
    cfg.DNN.FRCNN.TRAIN = edict()
    cfg.DNN.FRCNN.TRAIN.SELECTION_BATCH_SIZE = 32
    cfg.DNN.FRCNN.TRAIN.FG_RATIO = 0.5
    cfg.DNN.FRCNN.TRAIN.REMOVE_DIFFICULTS = False
    cfg.DNN.FRCNN.TRAIN.ADD_GTBOXES = True
    cfg.DNN.FRCNN.TRAIN.WEIGHTS = {'scores':1.0, 'rois':0.01}

    cfg.DNN.BIOMETRICS = edict()
    cfg.DNN.BIOMETRICS.BATCH_SIZE = 32

    cfg.BENCHMARK = edict()
    cfg.BENCHMARK.OVERLAP_THRESH = 0.5

    cfg.NETWORK = edict()

    cfg.NETWORK.FEAT_NAME = 'ZF'
    cfg.NETWORK.FEAT_INIT = None
    cfg.NETWORK.PREFIX = 'detector'

    cfg.NETWORK.PROPOSAL = edict()
    cfg.NETWORK.PROPOSAL.FEAT_NAME = {}
    cfg.NETWORK.PROPOSAL.FEAT_INIT = {}
    cfg.NETWORK.PROPOSAL.OPTIMIZER = {}
    cfg.NETWORK.PROPOSAL.NITER = 100000
    cfg.NETWORK.PROPOSAL.BATCH_SIZE = {}
    cfg.NETWORK.PROPOSAL.SELECTION_BATCH_SIZE = {}

    cfg.NETWORK.ANCHOR = edict()
    cfg.NETWORK.ANCHOR.FEAT_NAME = {}
    cfg.NETWORK.ANCHOR.FEAT_INIT = {}
    cfg.NETWORK.ANCHOR.OPTIMIZER = {}
    cfg.NETWORK.ANCHOR.NITER = 100000
    cfg.NETWORK.ANCHOR.BATCH_SIZE = {}
    cfg.NETWORK.ANCHOR.SELECTION_BATCH_SIZE = {}

    cfg.NETWORK.FRCNN = edict()
    cfg.NETWORK.FRCNN.FEAT_NAME = {}
    cfg.NETWORK.FRCNN.FEAT_INIT = {}
    cfg.NETWORK.FRCNN.OPTIMIZER = {}
    cfg.NETWORK.FRCNN.NITER = 100000
    cfg.NETWORK.FRCNN.BATCH_SIZE = {}
    cfg.NETWORK.FRCNN.SELECTION_BATCH_SIZE = {}

    return cfg
