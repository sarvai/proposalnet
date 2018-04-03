import tensorflow as tf
import numpy as np
import pickle

#from datasets_config import cfg as dset_cfg

class interface :
    def __init__( self, cfg ):
        self._cfg = cfg
        self._sections = {}

        self._inputs = {}
        self._nets = {}
        self._variables = {}

        self._prefix = self._cfg.dnn.NETWORK.PREFIX
        self._prefix_mod = self._cfg.dnn.NETWORK.PREFIX_MOD

        if self._prefix_mod is not None :
            self._prefix = "%s_%s" % ( self._prefix_mod, self._prefix )

        self._feat_name = self._cfg.dnn.NETWORK.FEAT_NAME

    def _build_variable_list( self, names ):
        variables = []
        tf_variables = tf.trainable_variables()

        for n in names :
            for tfv in tf_variables :
                if tfv.name.startswith( n ) :
                    variables.append( tfv )
        return variables

    def _get_all_variables( self ):
        vnames = []
        for key, item in self._sections.items() :
            vnames = vnames + item.names

        variables = self._build_variable_list( vnames )
        return variables

    def setup( self, deploy=False ):
        raise NotImplemented

    def eval( self, sess, blobs, net ):
        feed_dict = {}
        for k,item in net['inputs'].items() :
            if k in blobs :
                feed_dict[ item ] = blobs[ k ]

        out = sess.run( net['outputs'], feed_dict=feed_dict )
        return out

    def save( self, sess, path ):
        variables = self._get_all_variables()

        data = {}
        for v in variables :
            data[v.name] = v.eval(session=sess)

        with open(path,'wb') as ff :
            pickle.dump([ data ], ff )

    def load( self, sess, path ):
        variables = self._get_all_variables()

        var_map = {}
        var_check = {}
        for v in variables :
            var_map[v.name] = v
            var_check[v.name] = 0

        with open(path,'rb') as ff :
            data = pickle.load( ff )[0]

            for key,item in data.items() :
                if self._prefix_mod is not None :
                    key_mod = "%s_%s" % ( self._prefix_mod, key )
                else :
                    key_mod = key

                if item.shape != var_map[key_mod].get_shape() :
                    print( key_mod, item.shape, var_map[key_mod].get_shape() )
                sess.run( tf.assign( var_map[key_mod], item ) )
                var_check[key_mod] = 1

        unassigned = []
        for key, val in var_check.items() :
            if val == 0 :
                unassigned.append( key )

        if len( unassigned ) > 0 :
            print('Warning, the following variables were unassigned in loading')
            for u in unassigned :
                print( u )
