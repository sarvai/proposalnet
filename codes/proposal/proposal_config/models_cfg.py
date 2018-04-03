#

def get_proposal_structure( cfg ):
    if 'proposal' in cfg['NAME'] :
        proposal_structure = 'proposal'
    elif 'anchor' in cfg['NAME'] :
        proposal_structure = 'anchor'
    else :
        assert False, "Proposal structure not indentified"
    return proposal_structure

def end2end_s1( cfg, tag, data_hash, model_hash, mod=None ):
    models_cfg = {}
    assert( cfg.DATA is not None )

    proposal_structure = get_proposal_structure( cfg )

    models_cfg['niter'] = cfg.NETWORK.PROPOSAL.NITER
    models_cfg['feat_name'] = cfg.NETWORK.PROPOSAL.FEAT_NAME.get('s1',None)
    models_cfg['feat_init'] = cfg.NETWORK.PROPOSAL.FEAT_INIT.get('s1',None)
    models_cfg['optimizer'] = cfg.NETWORK.PROPOSAL.OPTIMIZER.get('s1',None)
    models_cfg['prefix'] = cfg.NETWORK.PREFIX
    models_cfg['batch_size'] = cfg.NETWORK.PROPOSAL.BATCH_SIZE.get('s1',None)
    models_cfg['selection_batch_size'] = cfg.NETWORK.FRCNN.SELECTION_BATCH_SIZE.get('s1',None)
    if tag is None :
        models_cfg['path'] = '%s/end2end_s1_%s_%s.pkl' % ( cfg.DATA.DIRS.MODELS,
                                                           models_cfg['feat_name'], model_hash )
        models_cfg['res'] = '%s/rois_end2end_s1_model_%s_dset_%s.pkl' % ( cfg.DATA.DIRS.DETS,
                                                                          model_hash, data_hash )
        models_cfg['benchmark'] = '%s/end2end_s1_model_%s_dset_%s_%s.pkl' % ( cfg.DATA.DIRS.BENCHMARK, model_hash,
                                                                              data_hash, '%s' )
    else :
        models_cfg['path'] = '%s/end2end_s1_%s_%s_%s.pkl' % ( cfg.DATA.DIRS.MODELS,
                                                              models_cfg['feat_name'], model_hash, tag )
        models_cfg['res'] = '%s/rois_end2end_s1_model_%s_dset_%s_%s.pkl' % ( cfg.DATA.DIRS.DETS,
                                                                             model_hash, data_hash, tag )
        models_cfg['benchmark'] = '%s/end2end_s1_model_%s_dset_%s_%s_%s.pkl' % ( cfg.DATA.DIRS.BENCHMARK, model_hash,
                                                                                 data_hash, tag, '%s' )
    models_cfg['rois'] = None
    models_cfg['pre'] = None
    models_cfg['use_negatives'] = True

    if mod is not None :
        for key, item in mod.items() :
            models_cfg[ key ] = item

    return models_cfg

models_cfg = {}
models_cfg['end2end_s1'] = end2end_s1
