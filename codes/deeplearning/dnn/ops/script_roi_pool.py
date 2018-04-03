import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import roi_pool
import tensorflow as tf

np.random.seed( 1234 )
flt_minf = np.finfo(np.float32).min

class test :
    def __init__( self ):
        self._pooled_height = 6
        self._pooled_width = 6
        self._spatial_scale = 1.0/16.0

        self._nbatches = 4
        self._frame_size = np.array([ 512, 512 ])
        self._nchannels = 12

    def gen_input( self ):
        self._input_shape = np.ceil( self._frame_size * self._spatial_scale ).astype(int)
        self._input = np.random.random(( self._nbatches, self._input_shape[0], self._input_shape[1], self._nchannels ))
        self._input = self._input.astype( np.float32 )

    def gen_rois( self ):
        h = self._frame_size[0]
        w = self._frame_size[1]

        self._rois = []

        for i in range( self._nbatches ):
            for j in range( 4 ):
                x = np.random.rand(2) * w
                y = np.random.rand(2) * h
                rr = [ i, np.min(x), np.min(y), np.max(x), np.max(y) ]
                self._rois.append( rr )

        self._rois = np.array( self._rois, dtype=np.float32 )
        self._nrois = len( self._rois )

        self._output_shape = [ self._nrois, self._pooled_height, self._pooled_width, self._nchannels ]

    def gen_grad( self ):
        self._grad_shape = self._output_shape
        self._grad = np.random.random( self._output_shape )

    def setup( self ):
        self.gen_input()
        self.gen_rois()
        self.gen_grad()
        self.get_network()

        self._sess = tf.InteractiveSession()
        self._sess.run(tf.global_variables_initializer())

    def get_network( self ):
        self._tensor_inputs = {}
        self._tensor_inputs['input'] = tf.placeholder( tf.float32, [ self._nbatches, self._input_shape[0], self._input_shape[1], self._nchannels ] )
        self._tensor_inputs['rois'] = tf.placeholder( tf.float32, [ self._nrois, 5 ] )
        self._tensor_inputs['grad'] = tf.placeholder( tf.float32, self._grad_shape )

        self._tensor_out = {}
        self._tensor_out['output'], self._tensor_out['argmax'] = roi_pool.roi_pool( self._tensor_inputs['input'],
                                                                self._tensor_inputs['rois'],
                                                                pooled_height=self._pooled_height,
                                                                pooled_width=self._pooled_width,
                                                                spatial_scale=self._spatial_scale)

        self._tensor_out['grad'] = roi_pool.grad_op( self._tensor_inputs['input'], self._tensor_out['argmax'], self._tensor_inputs['grad'] )

    def test_rois( self ):
        rois = np.round( self._rois[:,1:] * self._spatial_scale ).astype(int)
        width = rois[:,2] - rois[:,0] + 1
        height = rois[:,3] - rois[:,1] + 1

    def forward_python( self ):
        output = np.zeros((self._nrois, self._pooled_height, self._pooled_width, self._nchannels)).astype(np.float32)
        argmax = np.zeros((self._nrois, self._pooled_height, self._pooled_width, self._nchannels)).astype(np.int32)

        input_height, input_width = self._input.shape[1:3]
        nchannels = self._nchannels

        for index, x in np.ndenumerate( output ):
            n, ph, pw, c = index

            roi = self._rois[n]
            batch_index = int( roi[0] )
            roi = np.round( roi[1:] * self._spatial_scale )

            roi_width = np.max( [ roi[2] - roi[0] + 1.0, 1.0 ] )
            roi_height = np.max( [ roi[3] - roi[1] + 1.0, 1.0 ] )

            bin_size_h = float(roi_height) / self._pooled_height
            bin_size_w = float(roi_width) / self._pooled_width

            hstart = roi[1] + np.floor( float(ph) * bin_size_h )
            hstart = np.min( [ np.max( [ hstart , 0 ] ), input_height ] )

            wstart = roi[0] + np.floor( float(pw) * bin_size_w )
            wstart = np.min( [ np.max( [ wstart, 0 ] ), input_width ] )

            hend = roi[1] + np.ceil( float(ph+1) * bin_size_h )
            hend = np.min( [ np.max( [ hend , 0 ] ), input_height ] )

            wend = roi[0] + np.ceil( float(pw+1) * bin_size_w )
            wend = np.min( [ np.max( [ wend, 0 ] ), input_width ] )

            is_empty = ( hend <= hstart ) or ( wend <= wstart )

            maxval = flt_minf
            maxidx = -1

            if is_empty :
                maxval = 0

            for h in np.arange( hstart, hend ):
                for w in np.arange( wstart, wend ):
                    input_index = ( int(batch_index), int(h), int(w) , int(c) )

                    if self._input[ input_index ] > maxval :
                        maxval = self._input[ input_index ]
                        maxidx = ((int(batch_index)* input_height+ h)*input_width+w)*nchannels + c

            output[ index ] = maxval
            argmax[ index ] = maxidx

        return output, argmax

    def backward_python( self,argmax ):
        grad_out = np.zeros([ self._nbatches, self._input_shape[0], self._input_shape[1], self._nchannels ] )

        input_height, input_width = self._input.shape[1:3]
        nchannels = self._nchannels

        for index, x in np.ndenumerate( grad_out ):
            b,h,w,c = index
            gradient = 0.0

            for iter, roi_iter in enumerate(self._rois) :
                batch_index = int(roi_iter[0])
                if index[0] != int(roi_iter[0]):
                    continue

                roi = np.round( roi_iter[1:] * self._spatial_scale )
                in_roi = ( w >= roi[0] ) and ( h >= roi[1] ) and ( w <= roi[2] ) and ( h <= roi[3] )

                if not in_roi :
                    continue

                offset = iter * self._pooled_height * self._pooled_width * self._nchannels

                roi_width = np.max( [ roi[2] - roi[0] + 1.0, 1.0 ] )
                roi_height = np.max( [ roi[3] - roi[1] + 1.0, 1.0 ] )

                bin_size_h = float(roi_height) / self._pooled_height
                bin_size_w = float(roi_width) / self._pooled_width

                phstart = np.floor((h-roi[1])/bin_size_h)
                phstart = np.min( [ np.max([phstart,0]), self._pooled_height ] )

                phend = np.ceil((h-roi[1]+1)/bin_size_h)
                phend = np.min( [ np.max([phend,0]), self._pooled_height ] )

                pwstart = np.floor((w-roi[0])/bin_size_w)
                pwstart = np.min( [ np.max([pwstart,0]), self._pooled_width ] )

                pwend = np.ceil((w-roi[0]+1)/bin_size_w)
                pwend = np.min( [ np.max([pwend,0]), self._pooled_width ] )

                for ph in np.arange(phstart,phend) :
                    for pw in np.arange(pwstart,pwend) :
                        arg_index = ( iter, int(ph),int(pw), c )
                        if argmax[ arg_index ] == ((int(batch_index)* input_height+ h)*input_width+w)*nchannels + c :
                            gradient += self._grad[ arg_index ]

            grad_out[ index ] = gradient

        return grad_out

    def backward_python2( self, argmax ):
        grad_out = np.zeros([ self._nbatches, self._input_shape[0], self._input_shape[1], self._nchannels ] )

        input_height, input_width = self._input.shape[1:3]

        input_base_c = self._nchannels
        input_base_wc = input_width * input_base_c
        input_base_hwc = input_height * input_base_wc

        for index, input_index in np.ndenumerate( argmax ):
            if input_index >= 0 :
                c = int(input_index % self._nchannels)
                w = int(( input_index / input_base_c ) % input_width)
                h = int(( input_index / input_base_wc ) % input_height)
                n = int(( input_index / input_base_hwc ))
                grad_out[ (n,h,w,c) ] += self._grad[ index ]

        return grad_out

    def forward( self ):
        feed_dict = {}
        feed_dict[ self._tensor_inputs['input'] ] = self._input
        feed_dict[ self._tensor_inputs['rois'] ] = self._rois
        return self._sess.run( [ self._tensor_out['output'], self._tensor_out['argmax']], feed_dict=feed_dict )

    def forward_grad( self, argmax ):
        feed_dict = {}
        feed_dict[ self._tensor_inputs['input'] ] = self._input
        feed_dict[ self._tensor_out['argmax'] ] = argmax
        feed_dict[ self._tensor_inputs['grad'] ] = self._grad
        return self._sess.run( self._tensor_out['grad'], feed_dict=feed_dict )

    def python_grad_test( self ):
        py_out, py_argmax = self.forward_python()
        tf_out, tf_argmax = self.forward()

        py_grad = self.backward_python2( py_argmax )
        tf_grad = self.forward_grad( tf_argmax )

        check0 = np.linalg.norm( py_out - tf_out )
        check1 = np.linalg.norm( py_grad - tf_grad )

        print( check0, check1 )

if __name__=="__main__" :

    t = test()
    for i in range( 10 ):
        t.setup()
        t.python_grad_test()
