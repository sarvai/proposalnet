from .proposal import proposal_blobs
from .proposal import proposal_blobs_deploy
from .frcnn import frcnn_blobs, frcnn_blobs_hinge
from .frcnn import frcnn_blobs_deploy
from .frcnn import system_blobs_deploy
from .biometrics import biometrics_blobs
from .pose import pose_blobs

from .blobs import data_blobs

blobs_generator = {}
blobs_generator['proposal_s1'] = proposal_blobs
blobs_generator['end2end'] = proposal_blobs
blobs_generator['biometrics'] = biometrics_blobs
blobs_generator['pose'] = pose_blobs


blobs_generator_deploy = {}
blobs_generator_deploy['proposal_s1'] = proposal_blobs_deploy
blobs_generator_deploy['proposal_s1_video'] = system_blobs_deploy
blobs_generator_deploy['proposal_s2'] = proposal_blobs_deploy
blobs_generator_deploy['frcnn_s1'] = frcnn_blobs_deploy
blobs_generator_deploy['frcnn_s2'] = frcnn_blobs_deploy
blobs_generator_deploy['frcnn_s2_system'] = system_blobs_deploy
blobs_generator_deploy['frcnn_s2_video'] = system_blobs_deploy
blobs_generator_deploy['frcnn_s2_hinge'] = system_blobs_deploy
