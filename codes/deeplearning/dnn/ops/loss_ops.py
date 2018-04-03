import tensorflow as tf
from . import smooth_L1

def smooth_L1_loss( diffs, sigma=0.01 ):
	loss = smooth_L1.op( diffs, sigma=sigma )
	loss = tf.reduce_sum( loss, 1 )
	return loss
