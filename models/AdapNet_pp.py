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

import tensorflow as tf
import network_base
import numpy as np
import json
class AdapNet_pp(network_base.Network):
    def __init__(self, num_classes=12, learning_rate=0.001, float_type=tf.float32, weight_decay=0.0005,
                 decay_steps=30000, power=0.9, training=True, ignore_label=True, global_step=0,
                 has_aux_loss=True, model_def='default.json', mask = None):
        
        super(AdapNet_pp, self).__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.initializer = 'he'
        self.has_aux_loss = has_aux_loss
        self.float_type = float_type
        self.power = power
        self.decay_steps = decay_steps
        self.training = training
        self.bn_decay_ = 0.99
        self.eAspp_rate = [3, 6, 12]
        self.residual_units = [3, 4, 6, 3]
        with open(model_def) as f:
            tmp = json.load(f)
          
        self.filters = tmp['params'] #[256, 512, 1024, 2048]
        self.split_size = tmp['split']
        self.mask_id = None
        if mask is not None:
           self.mask_id = np.load(mask) 
        self.strides = [1, 2, 2, 1]
        self.global_step = global_step
        if self.training:
            self.keep_prob = 0.3
        else:
            self.keep_prob = 1.0
        if ignore_label:
            self.weights = tf.ones(self.num_classes-1)
            self.weights = tf.concat((tf.zeros(1), self.weights), 0)
        else:
            self.weights = tf.ones(self.num_classes)
     
    def _setup(self, data):
        self.input_shape = data.get_shape()
        with tf.variable_scope('conv0'):
            self.data_after_bn = self.batch_norm(data)
        tmp_cnt = 1
        self.conv_7x7_out = self.conv_batchN_relu(self.data_after_bn, 7, 2, self.filters[str(tmp_cnt)], name='conv1')
        self.max_pool_out = self.pool(self.conv_7x7_out, 3, 2)
        tmp_cnt += 1
        ##block1
        mask = None   
        if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
            mask = self.mask_id[()][str(tmp_cnt)]  
        self.m_b1_out = self.unit_0(self.max_pool_out, self.filters[str(tmp_cnt)], 1, 1, mask = mask)
        tmp_cnt += 1
        for unit_index in range(1, self.residual_units[0]):
            mask = None   
            if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
                mask = self.mask_id[()][str(tmp_cnt)]
            self.m_b1_out = self.unit_1(self.m_b1_out, self.filters[str(tmp_cnt)], 1, 1, unit_index+1, mask = mask)
            tmp_cnt += 1
        with tf.variable_scope('block1/unit_%d/bottleneck_v1/conv3'%self.residual_units[0]):
            self.b1_out = tf.nn.relu(self.batch_norm(self.m_b1_out))
        
        ##block2
        mask = None   
        if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
            mask = self.mask_id[()][str(tmp_cnt)] 
        self.m_b2_out = self.unit_1(self.b1_out, self.filters[str(tmp_cnt)], self.strides[1], 2, 1, shortcut=True, mask = mask)
        tmp_cnt += 1
        for unit_index in range(1, self.residual_units[1]-1):
            mask = None   
            if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
                mask = self.mask_id[()][str(tmp_cnt)]
            self.m_b2_out = self.unit_1(self.m_b2_out, self.filters[str(tmp_cnt)], 1, 2, unit_index+1, mask=mask)
            tmp_cnt += 1
        mask = None   
        if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
            mask = self.mask_id[()][str(tmp_cnt)]  
        self.m_b2_out = self.unit_3(self.m_b2_out, self.filters[str(tmp_cnt)], 2, self.residual_units[1], split_size=self.split_size[str(tmp_cnt)], mask=mask)
        tmp_cnt += 1
        with tf.variable_scope('block2/unit_%d/bottleneck_v1/conv3'%self.residual_units[1]):
            self.b2_out = tf.nn.relu(self.batch_norm(self.m_b2_out))

        ##block3
        mask = None   
        if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
            mask = self.mask_id[()][str(tmp_cnt)]
        self.m_b3_out = self.unit_1(self.b2_out, self.filters[str(tmp_cnt)], self.strides[2], 3, 1, shortcut=True, mask = mask)
        tmp_cnt += 1 
        mask = None   
        if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
            mask = self.mask_id[()][str(tmp_cnt)] 
        self.m_b3_out = self.unit_1(self.m_b3_out, self.filters[str(tmp_cnt)], 1, 3, 2, mask=mask)
        tmp_cnt += 1 
        for unit_index in range(2, self.residual_units[2]):
            mask = None   
            if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
                mask = self.mask_id[()][str(tmp_cnt)]
            self.m_b3_out = self.unit_3(self.m_b3_out, self.filters[str(tmp_cnt)], 3, unit_index+1, split_size=self.split_size[str(tmp_cnt)], mask=mask)
            tmp_cnt += 1
        ##block4
        mask = None   
        if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
            mask = self.mask_id[()][str(tmp_cnt)]
        self.m_b4_out = self.unit_4(self.m_b3_out, self.filters[str(tmp_cnt)], 4, 1, shortcut=True, split_size=self.split_size[str(tmp_cnt)], mask = mask)
        tmp_cnt += 1
        for unit_index in range(1, self.residual_units[3]):
            dropout = False
            if unit_index == 2:
                dropout = True
            mask = None   
            if self.mask_id is not None and str(tmp_cnt) in self.mask_id[()].keys():
                mask = self.mask_id[()][str(tmp_cnt)] 
            self.m_b4_out = self.unit_4(self.m_b4_out, self.filters[str(tmp_cnt)], 4, unit_index+1, dropout=dropout ,split_size=self.split_size[str(tmp_cnt)], mask=mask)
            tmp_cnt += 1 
        with tf.variable_scope('block4/unit_%d/bottleneck_v1/conv3'%self.residual_units[3]):
            self.b4_out = tf.nn.relu(self.batch_norm(self.m_b4_out))

        ##skip
        self.skip1 = self.conv_batchN_relu(self.b1_out, 1, 1, 24, name='conv32', relu=False)
        self.skip2 = self.conv_batchN_relu(self.b2_out, 1, 1, 24, name='conv174', relu=False)

        ##eAspp
        self.IA = self.conv_batchN_relu(self.b4_out, 1, 1, self.filters[str(tmp_cnt)], name='conv256')
        tmp_cnt += 1

        self.IB = self.conv_batchN_relu(self.b4_out, 1, 1, self.filters[str(tmp_cnt)], name='conv70')
        tmp_cnt += 1 
        self.IB = self.aconv_batchN_relu(self.IB, 3, self.eAspp_rate[0], self.filters[str(tmp_cnt)], name='conv7')
        tmp_cnt += 1
        self.IB = self.aconv_batchN_relu(self.IB, 3, self.eAspp_rate[0], self.filters[str(tmp_cnt)], name='conv247')
        tmp_cnt += 1 
        self.IB = self.conv_batchN_relu(self.IB, 1, 1, self.filters[str(tmp_cnt)], name='conv71')
        tmp_cnt += 1

        self.IC = self.conv_batchN_relu(self.b4_out, 1, 1, self.filters[str(tmp_cnt)], name='conv80')
        tmp_cnt += 1
        self.IC = self.aconv_batchN_relu(self.IC, 3, self.eAspp_rate[1], self.filters[str(tmp_cnt)], name='conv8')
        tmp_cnt += 1  
        self.IC = self.aconv_batchN_relu(self.IC, 3, self.eAspp_rate[1], self.filters[str(tmp_cnt)], name='conv248')
        tmp_cnt += 1
        self.IC = self.conv_batchN_relu(self.IC, 1, 1, self.filters[str(tmp_cnt)], name='conv81')
        tmp_cnt += 1

        self.ID = self.conv_batchN_relu(self.b4_out, 1, 1, self.filters[str(tmp_cnt)], name='conv90')
        tmp_cnt += 1
        self.ID = self.aconv_batchN_relu(self.ID, 3, self.eAspp_rate[2], self.filters[str(tmp_cnt)], name='conv9')
        tmp_cnt += 1
        self.ID = self.aconv_batchN_relu(self.ID, 3, self.eAspp_rate[2], self.filters[str(tmp_cnt)], name='conv249')
        tmp_cnt += 1
        self.ID = self.conv_batchN_relu(self.ID, 1, 1, self.filters[str(tmp_cnt)], name='conv91')
        tmp_cnt += 1

        self.IE = tf.expand_dims(tf.expand_dims(tf.reduce_mean(self.b4_out, [1, 2]), 1), 2)
        self.IE = self.conv_batchN_relu(self.IE, 1, 1, 256, name='conv57')
        self.IE_shape = self.b4_out.get_shape()
        self.IE = tf.image.resize_images(self.IE, [self.IE_shape[1], self.IE_shape[2]])

        self.eAspp_out = self.conv_batchN_relu(tf.concat((self.IA, self.IB, self.IC, self.ID, self.IE), 3), 1, 1, self.filters[str(tmp_cnt)], name='conv10', relu=False)
        tmp_cnt += 1
        ### Upsample/Decoder
        with tf.variable_scope('conv41'):
            self.deconv_up1 = self.tconv2d(self.eAspp_out, 4, self.filters[str(tmp_cnt)], 2)
            tmp_cnt += 1
            self.deconv_up1 = self.batch_norm(self.deconv_up1)

        self.up1 = self.conv_batchN_relu(tf.concat((self.deconv_up1, self.skip2), 3), 3, 1, self.filters[str(tmp_cnt)], name='conv89') 
        tmp_cnt += 1
        self.up1 = self.conv_batchN_relu(self.up1, 3, 1, self.filters[str(tmp_cnt)], name='conv96')
        tmp_cnt += 1
        with tf.variable_scope('conv16'):
            self.deconv_up2 = self.tconv2d(self.up1, 4, self.filters[str(tmp_cnt)], 2)
            tmp_cnt += 1
            self.deconv_up2 = self.batch_norm(self.deconv_up2)
        self.up2 = self.conv_batchN_relu(tf.concat((self.deconv_up2, self.skip1), 3), 3, 1, self.filters[str(tmp_cnt)], name='conv88')
        tmp_cnt += 1 
        self.up2 = self.conv_batchN_relu(self.up2, 3, 1, self.filters[str(tmp_cnt)], name='conv95')
        tmp_cnt += 1
        self.up2 = self.conv_batchN_relu(self.up2, 1, 1, self.num_classes, name='conv78')
        with tf.variable_scope('conv5'):
            self.deconv_up3 = self.tconv2d(self.up2, 8, self.num_classes, 4)
            self.deconv_up3 = self.batch_norm(self.deconv_up3)      

        self.softmax = tf.nn.softmax(self.deconv_up3)
        ## Auxilary
        if self.has_aux_loss:
            self.aux1 = tf.nn.softmax(tf.image.resize_images(self.conv_batchN_relu(self.deconv_up2, 1, 1, self.num_classes, name='conv911', relu=False), [self.input_shape[1], self.input_shape[2]]))
            self.aux2 = tf.nn.softmax(tf.image.resize_images(self.conv_batchN_relu(self.deconv_up1, 1, 1, self.num_classes, name='conv912', relu=False), [self.input_shape[1], self.input_shape[2]]))
        
        
    def _create_loss(self, label):
        self.loss = tf.reduce_mean(-tf.reduce_sum(tf.multiply(label*tf.log(self.softmax+1e-10), self.weights), axis=[3]))
        if self.has_aux_loss:
            aux_loss1 = tf.reduce_mean(-tf.reduce_sum(tf.multiply(label*tf.log(self.aux1+1e-10), self.weights), axis=[3]))
            aux_loss2 = tf.reduce_mean(-tf.reduce_sum(tf.multiply(label*tf.log(self.aux2+1e-10), self.weights), axis=[3]))
            self.loss = self.loss+0.6*aux_loss1+0.5*aux_loss2
    def create_optimizer(self):
        self.lr = tf.train.polynomial_decay(self.learning_rate, self.global_step,
                                            self.decay_steps, power=self.power)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss, global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            self.summary_op = tf.summary.merge_all()
    
    def build_graph(self, data, label=None, prunning=False):
        self._setup(data)
        if self.training or prunning:
            self._create_loss(label)

def main():
    print 'Do Nothing'
   
if __name__ == '__main__':
    main()

