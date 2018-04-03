import tensorflow as tf
import numpy as np

class network :
    def __init__( self, dnn_cfg, prefix=None ):
        self._dnn_cfg = dnn_cfg
        self._names = []
        self._groups = []
        self._prefix = prefix

    def reset( self ):
        self._names = []
        self._groups = []

    @property
    def names( self ):
        return self._names

    def names_group( self, gname ):
        groups = np.array( self._groups )
        names = np.array( self._names )
        inds = np.where( groups == gname )[0]

        return names[inds].tolist()

    def _scope_name( self, scope, group="" ):
        if self._prefix is None :
            name = scope
        else :
            name = '%s_%s' % ( self._prefix, scope )

        self._names.append( name )
        self._groups.append( group )

        return name

    def apply( self, inputs ):
        raise NotImplemented

    def loss( self, input, net ):
        raise NotImplemented

    def post_process( self, input, net ):
        raise NotImplemented

    def load_precalculated_weights( self, sess, path ):
        raise NotImplemented
