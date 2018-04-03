import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import smooth_L1
import tensorflow as tf
import sys

np.random.seed(1234)

def smooth_l1_forward( preds, targets, sigma=0.01 ):
	assert preds.shape == targets.shape

	data_shape = preds.shape
	sigma2 = sigma * sigma

	diff = preds - targets
	diff_abs = np.abs( preds - targets )

	diff = diff.ravel()
	diff_abs = diff_abs.ravel()

	out = np.zeros( len(diff) )

	for i in np.where( diff_abs < 1.0/sigma2 )[0] :
		out[i] = 0.5*diff[i]*diff[i]*sigma2

	for i in np.where( diff_abs >= 1.0/sigma2 )[0] :
		out[i] = diff_abs[i] - 0.5/sigma2

	return out.reshape( data_shape )

def smooth_l1_backward( preds, targets, grad, sigma=0.01 ):
	assert preds.shape == targets.shape
	assert preds.shape == grad.shape

	data_shape = preds.shape
	sigma2 = sigma * sigma

	diff = preds - targets
	diff_abs = np.abs( preds - targets )

	diff = diff.ravel()
	diff_abs = diff_abs.ravel()

	out = np.zeros( len(diff) )

	for i in np.where( diff_abs < 1.0/sigma2 ) :
		out[i] = diff[i]*sigma2

	for i in np.where( (diff_abs >= 1.0/sigma2) & ( diff > 0 ) )[0] :
		out[i] = 1.0

	for i in np.where( (diff_abs >= 1.0/sigma2) & ( diff < 0 ) )[0] :
		out[i] = -1.0

	out = out.reshape( data_shape )
	out = np.multiply(out, grad)

	return out

def test_gradient():
	preds = (np.random.random( [100,20] ) - 0.5) * 100
	targets = (np.random.random( [100,20] ) - 0.5 ) * 100
	grad = np.ones( [100,20], dtype=np.float32 )
	sigma = 0.001

	preds = preds.astype( np.float32 )
	targets = targets.astype( np.float32 )

	p = tf.placeholder( tf.float32, [100,20] )
	t = tf.placeholder( tf.float32, [100,20] )
	g = tf.placeholder( tf.float32, [100,20] )

	diff = p-t
	o = smooth_L1.op( diff, sigma=sigma )
	og = tf.gradients( o, [ p ] )[0]

	with tf.Session() as sess :
		out_tf = sess.run( o, feed_dict={diff:preds - targets} )
		out_py = smooth_l1_forward( preds, targets, sigma )
		print( np.linalg.norm( out_tf - out_py ) )

		out_grad_tf = sess.run( og, feed_dict={diff:preds - targets} )
		out_grad_py = smooth_l1_backward( preds, targets, grad, sigma)
		print( np.linalg.norm( out_grad_tf - out_grad_py ) )

def test_gradient_with_top():
	preds = (np.random.random( [100,20] ) - 0.5) * 100
	targets = (np.random.random( [100,20] ) - 0.5 ) * 100
	grad = np.ones( [100,20], dtype=np.float32 ) * 0.01
	sigma = 0.001

	preds = preds.astype( np.float32 )
	targets = targets.astype( np.float32 )

	p = tf.placeholder( tf.float32, [100,20] )
	t = tf.placeholder( tf.float32, [100,20] )
	g = tf.placeholder( tf.float32, [100,20] )

	diff = p-t
	o = smooth_L1.op( diff, sigma=sigma )

	o = tf.reduce_sum(o,1)
	o = tf.reduce_mean(o)

	og = tf.gradients( o, [ p ] )[0]

	with tf.Session() as sess :
		out_grad_tf = sess.run( og, feed_dict={diff:preds-targets, g:grad} )
		out_grad_py = smooth_l1_backward( preds, targets, grad, sigma)
		print( np.linalg.norm( out_grad_tf - out_grad_py ) )

if __name__=="__main__" :
	test_gradient()
	test_gradient_with_top()