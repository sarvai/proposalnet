import sys
import os
import argparse
import platform

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_commandline():
    parser = argparse.ArgumentParser(description='Detector Training')
    parser.add_argument('-d','--dataset',help='Dataset Information',required=True)
    parser.add_argument('-c','--config',help='Configuration Name',required=True)
    parser.add_argument('-m','--model',help='Model Name',required=True)
    parser.add_argument('-n','--nclasses',help='Number of object classes', default=1 )
    parser.add_argument('-b','--batchsize',help='Batch size', default=2 )
    parser.add_argument('-t','--tag',help='Data Tag', default=None)
    parser.add_argument('-p','--prefix',help='Model Prefix', default=None )
    parser.add_argument('--niter',help='Number of Iterations', default=None )
    args = parser.parse_args()

    model_params = args.model.split(',')

    params = {}
    params['cfg_name'] = args.config
    params['model_name'] = model_params[0]

    if len( model_params ) > 1 :
        params['model_params'] = model_params[1:]

    params['dset_params'] = args.dataset.split(',')
    params['nclasses'] = args.nclasses
    params['batch_size'] = args.batchsize
    params['niter'] = args.niter
    params['tag'] = args.tag
    params['prefix'] = args.prefix

    return params

def add_path( p ):
    if p not in sys.path :
        sys.path.append(p)

add_path('../common')
add_path('../cpp_common/build')
add_path('../deeplearning')
