import tensorflow as tf
import tensorflow.contrib.slim as slim
from .network import network

class feat_net0( network ):
    def apply( self, input ):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net = input-mean

        nets = {}

        with slim.arg_scope([slim.conv2d],activation_fn=tf.nn.relu,
                                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                        weights_regularizer=slim.l2_regularizer(0.05)) :
            net = slim.conv2d( net, 64, [3,3], scope=self._scope_name('feat_conv1') )
            net = slim.conv2d( net, 64, [3,3], scope=self._scope_name('feat_conv2') )
            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 128, [3,3], scope=self._scope_name('feat_conv3') )
            net = slim.conv2d( net, 128, [3,3], scope=self._scope_name('feat_conv4') )
            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 256, [3,3], scope=self._scope_name('feat_conv5') )



            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 256, [3,3], scope=self._scope_name('feat_conv6') )

            feat8 = {}
            feat8['net'] = net
            feat8['scale'] = 1.0/8.0
            feat8['base_size'] = 8.0

            nets['feat8'] = feat8

            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 512, [3,3], scope=self._scope_name('feat_conv7') )

            feat16 = {}
            feat16['net'] = net
            feat16['scale'] = 1.0/16.0
            feat16['base_size'] = 16.0

            nets['feat16'] = feat16

        return nets

class vgg16( network ):
    def apply( self, input ):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net = input-mean

        nets = {}

        with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.05)):
            net = slim.repeat( net, 2, slim.conv2d, 64, [3, 3], scope=self._scope_name('conv1'))
            net = slim.max_pool2d(net, [2, 2], scope=self._scope_name('pool1'))
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope=self._scope_name('conv2'))
            net = slim.max_pool2d(net, [2, 2], scope=self._scope_name('pool2'))
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope=self._scope_name('conv3'))

            feat4 = {}
            feat4['net'] = net
            feat4['scale'] = 1.0/4.0
            feat4['base_size'] = 4.0
            nets['feat4'] = feat4

            net = slim.max_pool2d(net, [2, 2], scope=self._scope_name('pool3'))
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope=self._scope_name('conv4'))

            feat8 = {}
            feat8['net'] = net
            feat8['scale'] = 1.0/8.0
            feat8['base_size'] = 8.0
            nets['feat8'] = feat8

            net = slim.max_pool2d(net, [2, 2], scope=self._scope_name('pool4'))
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope=self._scope_name('conv5'))

            feat16 = {}
            feat16['net'] = net
            feat16['scale'] = 1.0/16.0
            feat16['base_size'] = 16.0
            nets['feat16'] = feat16

        return nets

class vgg16_pose( network ):
    def apply( self, input ):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net = input-mean

        nets = {}

        with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.05)):
            net = slim.repeat( net, 2, slim.conv2d, 64, [3, 3], scope=self._scope_name('conv1'))
            #net = slim.max_pool2d(net, [2, 2], scope=self._scope_name('pool1'))
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope=self._scope_name('conv2'))
            #net = slim.max_pool2d(net, [2, 2], scope=self._scope_name('pool2'))
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope=self._scope_name('conv3'))

            #net = slim.max_pool2d(net, [2, 2], scope=self._scope_name('pool3'))
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope=self._scope_name('conv4'))

            feat1 = {}
            feat1['net'] = net
            feat1['scale'] = 1.0
            feat1['base_size'] = 1.0
            nets['feat8'] = feat1

            #net = slim.max_pool2d(net, [2, 2], scope=self._scope_name('pool4'))
            #net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope=self._scope_name('conv5'))

            #feat16 = {}
            #feat16['net'] = net
            #feat16['scale'] = 1.0/16.0
            #feat16['base_size'] = 16.0
            #nets['feat16'] = feat16

        return nets

class vgg16_small( network ):
    def apply( self, input ):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net = input-mean

        nets = {}

        with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.05)):
            net = slim.conv2d( net, 64, [3,3], scope=self._scope_name('feat_conv1') )
            net = slim.conv2d( net, 64, [3,3], scope=self._scope_name('feat_conv2') )
            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 128, [3,3], scope=self._scope_name('feat_conv3') )
            net = slim.conv2d( net, 128, [3,3], scope=self._scope_name('feat_conv4') )
            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 256, [3,3], scope=self._scope_name('feat_conv5') )

            feat4 = {}
            feat4['net'] = net
            feat4['scale'] = 1.0/4.0
            feat4['base_size'] = 4.0
            nets['feat4'] = feat4

            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 256, [3,3], scope=self._scope_name('feat_conv6') )

            feat8 = {}
            feat8['net'] = net
            feat8['scale'] = 1.0/8.0
            feat8['base_size'] = 8.0
            nets['feat8'] = feat8

            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 512, [3,3], scope=self._scope_name('feat_conv7') )

            feat16 = {}
            feat16['net'] = net
            feat16['scale'] = 1.0/16.0
            feat16['base_size'] = 16.0
            nets['feat16'] = feat16

        return nets

class vgg16_very_small( network ):
    def apply( self, input ):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net = input-mean

        nets = {}

        with slim.arg_scope([slim.conv2d],
                      activation_fn=tf.nn.relu,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.05)):
            net = slim.conv2d( net, 64, [3,3], scope=self._scope_name('feat_conv1') )
            net = slim.conv2d( net, 64, [3,3], scope=self._scope_name('feat_conv2') )
            net = slim.max_pool2d( net, [2,2] )
            net = slim.conv2d( net, 128, [3,3], scope=self._scope_name('feat_conv3') )
            net = slim.conv2d( net, 128, [3,3], scope=self._scope_name('feat_conv4') )

            feat2 = {}
            feat2['net'] = net
            feat2['scale'] = 1.0/2.0
            feat2['base_size'] = 2.0

            nets['feat2'] = feat2

        return nets

networks = {}
networks['feat_net0'] = feat_net0
networks['vgg16'] = vgg16
networks['vgg16_small'] = vgg16_small
networks['vgg16_very_small'] = vgg16_very_small
networks['vgg16_pose'] = vgg16_pose
