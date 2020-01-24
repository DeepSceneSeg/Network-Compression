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

import argparse
import datetime
import importlib
import os
import numpy as np
import pickle
import tensorflow as tf
import yaml
from dataset.helper import *

PARSER = argparse.ArgumentParser()
PARSER.add_argument('-c', '--config', default='config/cityscapes_test.config')

def test_func(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu_id']
    module = importlib.import_module('models.' + config['model'])
    model_func = getattr(module, config['model'])
    data_list, iterator = get_test_data(config)
    resnet_name = 'resnet_v2_50'

    with tf.variable_scope(resnet_name):
        model = model_func(num_classes=config['num_classes'], training=False ,mask=config['mask'], model_def=config['model_def'])
        images_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'], 3])
        labels_pl = tf.placeholder(tf.float32, [None, config['height'], config['width'],
                                                config['num_classes']])
        model.build_graph(images_pl, labels_pl, True)

    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())

    all_ops = tf.get_default_graph().get_operations()
    selected_ops = []
    name_of_ops = []
    selected_ops_grad = []
    
    for op in all_ops:
        if ('Conv2D' in op.name or 'conv2d_transpose' in op.name or 'convolution' in op.name) and ('conv2d_transpose/' not in op.name and 'convolution/' not in op.name and 'convolution_1/' not in op.name):
            print (op.name)
            selected_ops.append(op.outputs)
            name_of_ops.append(op.name)
            selected_ops_grad.append(tf.gradients(model.loss, op.outputs))
    print ('total_number_of_operations: ', len(name_of_ops))

    import_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print 'total_variables_loaded:', len(import_variables)
    saver = tf.train.Saver(import_variables)
    saver.restore(sess, config['checkpoint'])
    sess.run(iterator.initializer)
    step = 0
    total_num = 0
    output_matrix = np.zeros([config['num_classes'], 3])
    rank_values={}
    start = True
    while 1:
        try:
            img, label = sess.run([data_list[0], data_list[1]])
            feed_dict = {images_pl : img, labels_pl: label}
            activations = sess.run(selected_ops, feed_dict=feed_dict)
            gradients = sess.run(selected_ops_grad, feed_dict=feed_dict)
            total_num += label.shape[0]
            i = 0
            for activation, gradient in zip(activations,gradients):
                v=np.sum(activation[0]*gradient[0], axis=(0,1,2))
                v=v/(activation[0].shape[0]*activation[0].shape[1]*activation[0].shape[2])
                if start == True:
                    rank_values[name_of_ops[i]] = v
                else:
                    rank_values[name_of_ops[i]] = rank_values[name_of_ops[i]]+v
                i = i+1
            if (step+1) % config['skip_step'] == 0:
                print '%s %s] %d. nvidia rank updating' \
                  % (str(datetime.datetime.now()), str(os.getpid()), total_num)
            step += 1
            start = False
            
        except tf.errors.OutOfRangeError:
            f=open(config['rank_save'],'wb')
            pickle.dump(rank_values,f)
            f.close()
            print 'done'
            break

def main():
    args = PARSER.parse_args()
    if args.config:
        file_address = open(args.config)
        config = yaml.load(file_address)
    else:
        print '--config config_file_address missing'
    test_func(config)

if __name__ == '__main__':
    main()
