import importlib
import json
import numpy as np
import os
import pickle
import re
import tensorflow as tf
from matplotlib import pyplot as plt
from models.mapping import mapping

decide_threshold = False # to visualize the histogram of l1 norm
phase = 'decoder_conv1'  #conv1, block1/unit_{1,2,3}, block2/unit_{1,2,3,4}, block3/unit_{1,2,3,4,5,6}, block4unit_{1,2,3}, easpp, upsample1, upsample2, decoder_conv1, decoder_conv2
                  # upsample1 is the first deconvolution layer w.r.t input side of the network followed by upsample2 and upsample3
                  # decoder_conv1 is the 3x3 conv layer pair between upsample1 and upsample2 layers
                  # decoder_conv2 is the 3x3 conv layer pair between upsample2 and upsample3 layers  

all_convs = False #set true, to run for all convolutions layers under a given phase. Pruning layer by layer is recommended rather than all at once.

conv_name = 'conv1' #in case of block phase, conv1, conv2, conv3, if the block consists of multiphase than conv2_convolution (block 3 and 4), conv2_convolution_1 (block 4)
                    #in case of decoder_convs, conv1 and conv2
                    #in case of easpp:
                    #                 'conv256' 1x1
                    #                 'conv70' 1x1 or 'conv7' 3x3 first atrous or 'conv247' 3x3 second atrous or 'conv71' 1x1 rate=3 unit  
                    #                 'conv80' 1x1 or 'conv8' 3x3 first atrous or 'conv248' 3x3 second atrous or 'conv81' 1x1 rate=6 unit  
                    #                 'conv90' 1x1 or 'conv9' 3x3 first atrous or 'conv249' 3x3 second atrous or 'conv91' 1x1 rate=12 unit
                    #                 'conv10' final 1x1  
                    
save_nvidia_name = 'n1.p' # path to the nvidia computed rank values
list_of_convs = ['conv256/', 'conv70/', 'conv80/', 'conv90/', 'conv10/', 'conv71/', 'conv81/', 'conv91/', 'conv7/', 'conv8/', 'conv9/', 'conv247/', 'conv248/', 'conv249/']
threshold = 0.2
checkpoint_name = '/home/mohan/AdapNet_Training/checkpoint_final_cityuni/adapnet_sc-121999' # path address to checkpoint
gpu_id = '1'
try_zeros = False
new_checkpoint_save = 'check/adapnet_sc-122999' #path to prunned one
mask_load = None # incase mask already exist
mask_save = 'temp.npy' # saving mask
Num_classes = 12
height = 384
width = 768
model_def='models/default.json' #set it to the current model definition
new_model_def='models/1.json' #path to new model definition


def get_l1_norm(x):
    x[x<0] = 0.0            
    return np.abs(x)/np.max(x)

def get_mask_id(x):
    p=np.where(x==0)
    p=p[0].reshape(-1,1)
    p=np.int32(p)
    return p

f = open(save_nvidia_name)
rank_values = pickle.load(f)
compute =  False
trimmed = []

for op_name in rank_values:
    if phase == 'conv1':
       if phase+'/' in op_name and 'block' not in op_name:
           compute = True
    
    elif 'block' in phase:
       if phase+'/' in op_name:
           if all_convs:
               compute = True
           else:
               if conv_name+'/' in op_name:
                   compute = True
               elif len(conv_name.split('_')) == 2:
                   parts = conv_name.split('_')
                   if parts[0]+'/' in op_name and parts[1] in op_name:
                       compute = True
               else:
                   parts = conv_name.split('_')
                   if parts[0]+'/' in op_name and '_'.join(parts[:1]) in op_name:
                       compute = True

    elif 'easpp' in phase:
        if all_convs:
            for some_conv in list_of_convs:
               if some_conv in op_name:
                   compute = True
        elif conv_name+'/' in op_name and conv_name+'/' in list_of_convs:
               compute = True    

    elif 'decoder_conv1' in phase:
        if ('conv1' in conv_name or all_convs) and 'conv89/' in op_name: 
            compute = True
        elif ('conv2' in conv_name or all_convs) and 'conv96/' in op_name: 
            compute = True
    
    elif 'decoder_conv2' in phase:
        if ('conv1' in conv_name or all_convs) and 'conv88/' in op_name: 
            compute = True
        elif ('conv2' in conv_name or all_convs) and 'conv95/' in op_name: 
            compute = True    
                
    elif 'upsample1' in phase and 'conv41/' in op_name:
        compute = True
    
    elif 'upsample2' in phase and 'conv16/' in op_name:
        compute = True      

    if compute:
        print op_name
        l1 = get_l1_norm(rank_values[op_name])
        sorted_ = np.argsort(l1)
        norm_val_sorted = l1[sorted_]
        if decide_threshold:
            plt.hist(norm_val_sorted)
            plt.show()
        else:
            mask = l1<threshold
            trimmed.append([op_name, mask])         
    
    compute = False
 
if decide_threshold == False:
    with open(model_def) as f:
        model_definition = json.load(f)
    mask_id = {}
    if mask_load is not None:
        mask_id = np.load(mask_load)[()]
        
    reader=tf.train.NewCheckpointReader(checkpoint_name)
    weights_str = reader.debug_string()
    exclude_variables = {}
    tensor_list = []
    mask_exist = False
    for trim in trimmed:
        if try_zeros:
            name = ('/').join(trim[0].split('/')[:-1])+'/weights'
            if name not in exclude_variables:     
                tensor = reader.get_tensor(name)
            else:
                tensor = exclude_variables[name]
            mask = trim[1]
            
            if 'split' in mapping[trim[0]].keys() and mapping[trim[0]]['split'] == 2:
                mask = np.concatenate((np.zeros(model_definition['split'][mapping[trim[0]]['id']][0],dtype = mask.dtype), mask), -1)
                    
            elif 'split' in mapping[trim[0]].keys() and mapping[trim[0]]['split'] == 1:
                mask = np.concatenate((mask, np.zeros(model_definition['split'][mapping[trim[0]]['id']][1],dtype = mask.dtype)), -1)

            tensor[:,:,:,mask] = 0.0
            exclude_variables[name] = tensor 
                        
        else:
            mask = trim[1]
                
            if 'split' in mapping[trim[0]].keys() and mapping[trim[0]]['split'] == 2:
                temp = mask.copy()
                mask = np.concatenate((np.zeros(model_definition['split'][mapping[trim[0]]['id']][0],dtype = mask.dtype), mask), -1)
                model_definition['split'][mapping[trim[0]]['id']][1] = np.sum(temp==0)
            elif 'split' in mapping[trim[0]].keys() and mapping[trim[0]]['split'] == 1:
                temp = mask.copy()
                mask = np.concatenate((mask, np.zeros(model_definition['split'][mapping[trim[0]]['id']][1],dtype = mask.dtype)), -1)
                model_definition['split'][mapping[trim[0]]['id']][0] = np.sum(temp==0)  
            if mapping[trim[0]]['BatchNorm']:
                stuffs = ['/weights','/BatchNorm/beta','/BatchNorm/gamma','/BatchNorm/moving_mean','/BatchNorm/moving_variance']
            else:
                stuffs = ['/weights']

            for stuff in stuffs:
                name = ('/').join(trim[0].split('/')[:-1])+stuff
                if name not in exclude_variables:     
                    tensor = reader.get_tensor(name)
                else:
                    tensor = exclude_variables[name]

                if stuff == '/weights' and 'transpose' not in trim[0]:
                    tensor=np.delete(tensor,np.argwhere(mask==1),3)
                elif stuff == '/weights' and 'transpose' in trim[0]:
                    tensor=np.delete(tensor,np.argwhere(mask==1),2)
                else:
                    tensor=np.delete(tensor,np.argwhere(mask==1),0)
                exclude_variables[name] = tensor
                
            if 'place' not in mapping[trim[0]].keys():
                model_definition['params'][mapping[trim[0]]['id']] = tensor.shape[-1]
            else: 
                model_definition['params'][mapping[trim[0]]['id']][mapping[trim[0]]['place']] = tensor.shape[-1]
            if 'mask' in mapping[trim[0]].keys():
                    mask_id[mapping[trim[0]]['id']] = get_mask_id(mask)
                    mask_exist = True
  
            for key in mapping[trim[0]]:
                if 'next' in key:
                    name = ('/').join(mapping[trim[0]][key].split('/')[:-1])+'/weights'     
                    if name not in exclude_variables:     
                        tensor = reader.get_tensor(name)
                    else:
                        tensor = exclude_variables[name]
                    
                    if 'transpose' in mapping[trim[0]][key]:
                        tensor=np.delete(tensor,np.argwhere(mask==1),3)
                    else:
                        tensor=np.delete(tensor,np.argwhere(mask==1),2)
                    exclude_variables[name] = tensor
                    

    if try_zeros == False:
       with open(new_model_def, 'w') as f:
           json.dump(model_definition, f)
       model_def = new_model_def
    if mask_exist:
       with open(mask_save, 'w') as f:
           np.save(f, mask_id)
    else:
       mask_save = None

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    module = importlib.import_module('models.' + 'AdapNet_pp')
    model_func = getattr(module, 'AdapNet_pp')
    resnet_name = 'resnet_v2_50'

    with tf.variable_scope(resnet_name):
        model = model_func(num_classes=Num_classes, training=False, model_def=model_def, mask=mask_save)
        images_pl = tf.placeholder(tf.float32, [None, height, width, 3])
        model.build_graph(images_pl)

    config1 = tf.ConfigProto()
    config1.gpu_options.allow_growth = True
    sess = tf.Session(config=config1)
    sess.run(tf.global_variables_initializer())
    all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    load_var = {}
    index_record = {}
    for i,variable in enumerate(all_variables):
        name_ = variable.name.split(':')[0] 
        if name_ not in exclude_variables:
            load_var[name_] = variable
        else:
            index_record[name_] = i
    saver = tf.train.Saver(load_var)
    saver.restore(sess, checkpoint_name)
    
    for name_ in exclude_variables:
        print name_+':0'
        process = all_variables[index_record[name_]].assign(exclude_variables[name_])
        _=sess.run(process)

    saver = tf.train.Saver()
    saver.save(sess, new_checkpoint_save)    

