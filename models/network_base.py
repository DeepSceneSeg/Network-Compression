''' AdapNet++:  Self-Supervised Model Adaptation for Multimodal Semantic Segmentation

 Copyright (C) 2018  Abhinav Valada, Rohit Mohan and Wolfram Burgard

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.'''

import numpy as np
import tensorflow as tf

class Network(object):
    def __init__(self):
        print 'Network_Construction'

    def _setup(self, data):
        raise NotImplementedError("Implement this method.")

    def _create_loss(self, label):
        raise NotImplementedError("Implement this method.")

    def _create_optimizer(self):
        raise NotImplementedError("Implement this method.")

    def _create_summaries(self):
        raise NotImplementedError("Implement this method.")

    def build_graph(self, data, label=None):
        raise NotImplementedError("Implement this method.")

    def conv2d(self, inputs, kernel_size, stride, out_channels, name=None, padding='SAME'):
        in_channels = inputs.get_shape().as_list()[-1]
        weight_shape = [kernel_size, kernel_size, in_channels, out_channels]
        initializer = tf.contrib.layers.xavier_initializer()

        if self.initializer is 'he':
            n = kernel_size * kernel_size * in_channels
            std = np.sqrt(2.0 / n)
            initializer = tf.truncated_normal_initializer(stddev=std)

        if name is None:
            name = 'weights'

        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            kernel = tf.get_variable('', weight_shape,
                                     initializer=initializer,
                                     trainable=self.training,
                                     dtype=self.float_type,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
                                    )
        return tf.nn.conv2d(inputs, kernel, strides=strides, padding=padding)

    def split_conv2d(self, inputs, kernel_size, rate, out_channels, name=None, padding='SAME', both_atrous=False, split_size=None):
        in_channels = inputs.get_shape().as_list()[-1]
        weight_shape = [kernel_size, kernel_size, in_channels, out_channels]
        initializer = tf.contrib.layers.xavier_initializer()
        if self.initializer is 'he':
            n = kernel_size * kernel_size * in_channels
            std = np.sqrt(2.0 / n)
            initializer = tf.truncated_normal_initializer(stddev=std)
        if name is None:
            name = 'weights'
      
        strides = [1, 1, 1, 1]
        with tf.variable_scope(name):
            kernel = tf.get_variable('', weight_shape,
                                     initializer=initializer,
                                     trainable=self.training,
                                     dtype=self.float_type,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
                                    )
        if split_size is None:
            kernelA, kernelB = tf.split(kernel, 2, 3)
        else:
            kernelA, kernelB = tf.split(kernel, split_size, 3)  
        if both_atrous:
            outA = tf.nn.atrous_conv2d(inputs, kernelA, rate, padding=padding)
            outB = tf.nn.atrous_conv2d(inputs, kernelB, rate, padding=padding)
        else:
            outA = tf.nn.conv2d(inputs, kernelA, strides=strides, padding=padding)
            outB = tf.nn.atrous_conv2d(inputs, kernelB, rate, padding=padding)
        return tf.concat((outA, outB), 3)

    def batch_norm(self, inputs):
        in_channels = inputs.get_shape().as_list()[-1]

        with tf.variable_scope('BatchNorm'):
            gamma = tf.get_variable('gamma', (in_channels,), initializer=tf.constant_initializer(1.0),
                                    trainable=self.training, dtype=self.float_type,
                                    regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))
            beta = tf.get_variable('beta', (in_channels,), initializer=tf.constant_initializer(0),
                                   trainable=self.training, dtype=self.float_type,
                                   regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))
            moving_mean = tf.get_variable('moving_mean', (in_channels,),
                                          initializer=tf.constant_initializer(0), trainable=False, dtype=self.float_type)
            moving_var = tf.get_variable('moving_variance', (in_channels,),
                                         initializer=tf.constant_initializer(1), trainable=False, dtype=self.float_type)

        if self.training:
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
            train_mean = tf.assign(moving_mean, moving_mean* self.bn_decay_ + batch_mean * (1 - self.bn_decay_))
            train_var = tf.assign(moving_var, moving_var * self.bn_decay_ + batch_var * (1 - self.bn_decay_))
            with tf.control_dependencies([train_mean, train_var]):
                return  tf.nn.batch_normalization(inputs, batch_mean, batch_var, scale=gamma, offset=beta, variance_epsilon=0.00001)
        else:
            return  tf.nn.batch_normalization(inputs, moving_mean, moving_var, scale=gamma, offset=beta, variance_epsilon=0.00001)

    def pool(self, x, k_size, strides, name=None, padding='SAME'):
        return tf.nn.max_pool(x, [1, k_size, k_size, 1], strides=[1, strides, strides, 1], padding=padding)

    def atrous(self, inputs, kernel_size, rate, out_channels, name=None, padding='SAME'):
        in_channels = inputs.get_shape().as_list()[-1]
        weight_shape = [kernel_size, kernel_size, in_channels, out_channels]
        initializer = tf.contrib.layers.xavier_initializer()

        if self.initializer is 'he':
            n = kernel_size * kernel_size * in_channels
            std = np.sqrt(2.0 / n)
            initializer = tf.truncated_normal_initializer(stddev=std)

        if name is None:
            name = 'weights'

        with tf.variable_scope(name):
            kernel = tf.get_variable('', weight_shape,
                                     initializer=initializer,
                                     trainable=self.training,
                                     dtype=self.float_type,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
                                    )
        return tf.nn.atrous_conv2d(inputs, kernel, rate, padding=padding)

    def tconv2d(self, inputs, kernel_size, out_channels, stride, name=None, padding='SAME'):
        in_channels = inputs.get_shape().as_list()[-1]
        weight_shape = [kernel_size, kernel_size, out_channels, in_channels]
        n = kernel_size * kernel_size * in_channels
        std = np.sqrt(2.0 / n)

        if name is None:
            name = 'weights'

        initializer = tf.truncated_normal_initializer(stddev=std)
        with tf.variable_scope(name):
            kernel = tf.get_variable('', weight_shape,
                                     initializer=initializer,
                                     trainable=self.training,
                                     dtype=self.float_type,
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay)
                                    )

        in_shape = tf.shape(inputs)
        h = in_shape[1]*stride
        w = in_shape[2]*stride 
        new_shape = [in_shape[0], h, w, out_channels]
        return tf.nn.conv2d_transpose(inputs, kernel, new_shape, strides=[1, stride, stride, 1], padding=padding)

    def conv_batchN_relu(self, x, kernel_size, stride, out_channels, name, relu=True, dropout=False):
        with tf.variable_scope(name):
            conv_out = self.conv2d(x, kernel_size, stride, out_channels)
            if dropout:
                conv_out = tf.nn.dropout(conv_out, self.keep_prob)
            conv_batch_norm_out = self.batch_norm(conv_out)
            if relu:
                conv_activation = tf.nn.relu(conv_batch_norm_out)
                return conv_activation
            else:
                return conv_batch_norm_out

    def aconv_batchN_relu(self, x, kernel_size, rate, out_channels, name, relu=True):
        with tf.variable_scope(name):
            conv_out = self.atrous(x, kernel_size, rate, out_channels)
            conv_batch_norm_out = self.batch_norm(conv_out)
            if relu:
                conv_activation = tf.nn.relu(conv_batch_norm_out)
                return conv_activation
            else:
                return conv_batch_norm_out

    def unit_0(self, x, filter_num, block, unit, mask=None):
        with tf.variable_scope('block%d/unit_%d/bottleneck_v1'%(block, unit)):
            b_u_1r_cout = self.conv_batchN_relu(x, 1, 1, filter_num[0], name='conv1')
            b_u_3_cout = self.conv_batchN_relu(b_u_1r_cout, 3, 1, filter_num[1], name='conv2')
            with tf.variable_scope('conv3'):
                b_u_1e_cout=self.conv2d(b_u_3_cout, 1, 1, filter_num[2])
                if mask is not None:
                    inp=tf.shape(b_u_1e_cout)
        	    idx=tf.constant(mask)
                    update=tf.transpose(b_u_1e_cout,[3,2,1,0])
                    scatter_out=tf.scatter_nd(idx,update,[filter_num[3],inp[2],inp[1],inp[0]])
                    b_u_1e_cout=tf.transpose(scatter_out,[3,2,1,0])
                    
            with tf.variable_scope('shortcut'):
                b_u_s_cout = self.conv2d(b_u_1r_cout, 1, 1, filter_num[3])
            
            b_u_out = tf.add(b_u_1e_cout, b_u_s_cout)
        return b_u_out    
    
    def unit_1(self, x, filter_num, stride, block, unit, shortcut=False, mask = None):
        if shortcut == False:
            with tf.variable_scope('block%d/unit_%d/bottleneck_v1/conv3'%(block, unit-1)):
                x_bn = tf.nn.relu(self.batch_norm(x))
        else:
                x_bn = x
        with tf.variable_scope('block%d/unit_%d/bottleneck_v1'%(block, unit)):
            b_u_1r_cout = self.conv_batchN_relu(x_bn, 1, 1, filter_num[0], name='conv1')
            b_u_3_cout = self.conv_batchN_relu(b_u_1r_cout, 3, stride, filter_num[1], name='conv2')
            with tf.variable_scope('conv3'):
                b_u_1e_cout = self.conv2d(b_u_3_cout, 1, 1, filter_num[2])
                if mask is not None:
                    inp=tf.shape(b_u_1e_cout)
        	    idx=tf.constant(mask)
                    update=tf.transpose(b_u_1e_cout,[3,2,1,0])
                    if shortcut:
                        scatter_out=tf.scatter_nd(idx,update,[filter_num[3],inp[2],inp[1],inp[0]])
                    else:
                        scatter_out=tf.scatter_nd(idx,update,[x.shape[3],inp[2],inp[1],inp[0]])
                    b_u_1e_cout=tf.transpose(scatter_out,[3,2,1,0])
            if shortcut:
                with tf.variable_scope('shortcut'):
                    b_u_s_cout = self.conv2d(x_bn, 1, stride, filter_num[3])
                    b_u_out = tf.add(b_u_1e_cout, b_u_s_cout)
            else:
                b_u_out = tf.add(b_u_1e_cout, x)
             
            return b_u_out

    def unit_3(self, x, filter_num, block, unit, split_size=None, mask=None):
        with tf.variable_scope('block%d/unit_%d/bottleneck_v1/conv3'%(block, unit-1)):
            x_bn = tf.nn.relu(self.batch_norm(x))
        with tf.variable_scope('block%d/unit_%d/bottleneck_v1'%(block, unit)):
            b_u_1r_cout = self.conv_batchN_relu(x_bn, 1, 1, filter_num[0], name='conv1')
            with tf.variable_scope('conv2'):
                b3_u3_3_cout = self.split_conv2d(b_u_1r_cout, 3, 2, filter_num[1], split_size=split_size)
                b3_u3_3_bnout = self.batch_norm(b3_u3_3_cout)
                b3_u3_3_ract = tf.nn.relu(b3_u3_3_bnout)
                if mask is not None:
                    inp=tf.shape(b_u_1e_cout)
        	    idx=tf.constant(mask)
                    update=tf.transpose(b3_u3_1e_cout,[3,2,1,0])
                    scatter_out=tf.scatter_nd(idx,update,[x.shape[3],inp[2],inp[1],inp[0]])
                    b3_u3_1e_cout=tf.transpose(scatter_out,[3,2,1,0])
            with tf.variable_scope('conv3'):
                b3_u3_1e_cout = self.conv2d(b3_u3_3_ract, 1, 1, filter_num[2])
                 
            b3_u3_out = tf.add(x, b3_u3_1e_cout)

        return b3_u3_out

    def unit_4(self, x, filter_num, block, unit, shortcut=False, dropout=False, split_size=None, mask = None):
        if shortcut:
            with tf.variable_scope('block%d/unit_%d/bottleneck_v1/conv3'%(block-1, 6)):
                x_bn = tf.nn.relu(self.batch_norm(x))
        else:
            with tf.variable_scope('block%d/unit_%d/bottleneck_v1/conv3'%(block, unit-1)):
                x_bn = tf.nn.relu(self.batch_norm(x))

        with tf.variable_scope('block%d/unit_%d/bottleneck_v1'%(block, unit)):
            b_u_1r_cout = self.conv_batchN_relu(x_bn, 1, 1, filter_num[0], name='conv1')
            with tf.variable_scope('conv2'):
                b3_u3_3_cout = self.split_conv2d(b_u_1r_cout, 3, 2, filter_num[1], both_atrous=True, split_size=split_size)
                b3_u3_3_bnout = self.batch_norm(b3_u3_3_cout)
                b3_u3_3_ract = tf.nn.relu(b3_u3_3_bnout)

            with tf.variable_scope('conv3'):
                b3_u3_1e_cout=self.conv2d(b3_u3_3_ract, 1, 1, filter_num[2])
                if mask is not None:
                    inp=tf.shape(b_u_1e_cout)
        	    idx=tf.constant(mask)
                    update=tf.transpose(b3_u3_1e_cout,[3,2,1,0])
                    if shortcut:
                        scatter_out=tf.scatter_nd(idx,update,[filter_num[3],inp[2],inp[1],inp[0]])
                    else:
                        scatter_out=tf.scatter_nd(idx,update,[x.shape[3],inp[2],inp[1],inp[0]])
                    b3_u3_1e_cout=tf.transpose(scatter_out,[3,2,1,0])
            if dropout:
        	b3_u3_1e_cout=tf.nn.dropout(b3_u3_1e_cout, self.keep_prob)

            if shortcut:
                with tf.variable_scope('shortcut'):
                    b_u_s_cout = self.conv2d(x_bn, 1, 1, filter_num[3])
                    b3_u3_out = tf.add(b_u_s_cout, b3_u3_1e_cout)
            else:
                b3_u3_out = tf.add(x, b3_u3_1e_cout)

        return b3_u3_out

    def fc(self, inputs, out_channels, name):
        in_channels = inputs.get_shape().as_list()[-1]
        weight_shape = [in_channels, out_channels]
        initializer = tf.contrib.layers.xavier_initializer()
        if self.initializer is 'he':
            n = in_channels
            std = np.sqrt(2.0 / n)
            initializer = tf.truncated_normal_initializer(stddev=std)
        name_W = name+'/weights'
        
        with tf.variable_scope(name_W):
            kernel = tf.get_variable('', weight_shape,
                                     initializer=initializer,
                                     trainable=self.training,
                                     dtype=self.float_type, 
                                     regularizer=tf.contrib.layers.l2_regularizer(scale=self.weight_decay))
            x = tf.matmul(inputs, kernel)
        with tf.variable_scope(name): 
            b = tf.get_variable('biases', [out_channels], trainable=self.training,
                                initializer=tf.constant_initializer(0.01), dtype=self.float_type)
        return tf.nn.bias_add(x, b, data_format='NHWC')

    def conv_bias(self, inputs, kernel_size, stride, out_channels, name):
        with tf.variable_scope(name):
            x = self.conv2d(inputs, kernel_size, stride, out_channels)
            b = tf.get_variable('biases', [out_channels], trainable=self.training,
                                initializer=tf.constant_initializer(0.01), dtype=self.float_type)
        return tf.nn.bias_add(x, b, data_format='NHWC')
