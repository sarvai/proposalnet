from easydict import EasyDict as edict

from .network import network
from .proposal_network import proposal_network
from .extended_proposal_network import extended_proposal_network
from .frcnn_network import frcnn_network
from .feat_networks import networks as feature_networks
from .biometrics_network import biometrics_network
from .categories_network import categories_network
from .pose_network import pose_network
from . import selection_networks

networks = edict()
networks.proposal = proposal_network
networks.extended_proposal = extended_proposal_network
networks.frcnn = frcnn_network
networks.feature = feature_networks
networks.biometrics = biometrics_network
networks.categories = categories_network
networks.pose = pose_network

networks.selection = edict()
networks.selection.flat = selection_networks.flat
networks.selection.roipool = selection_networks.roipool
