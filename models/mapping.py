mapping = {
'resnet_v1_50/conv1/Conv2D':{'next': 'resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/Conv2D', 'id': '1', 'BatchNorm': True},
'resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/Conv2D', 'id': '2', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block1/unit_1/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/Conv2D', 'id': '2', 'place': 1, 'BatchNorm': True},
'resnet_v1_50/block1/unit_1/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '2', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block1/unit_2/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/Conv2D', 'id': '3', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block1/unit_2/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/Conv2D', 'id': '3', 'place': 1, 'BatchNorm': True},
'resnet_v1_50/block1/unit_2/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '3', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block1/unit_3/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Conv2D', 'id': '4', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block1/unit_3/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/Conv2D', 'id': '4', 'place': 1, 'BatchNorm': True},
'resnet_v1_50/block1/unit_3/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '4', 'place': 2, 'BatchNorm': False},

'resnet_v1_50/block2/unit_1/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/Conv2D', 'id': '5', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block2/unit_1/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/Conv2D', 'id': '5', 'place': 1, 'BatchNorm': True},
'resnet_v1_50/block2/unit_1/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '5', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block2/unit_2/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/Conv2D', 'id': '6', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block2/unit_2/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/Conv2D', 'id': '6', 'place': 1, 'BatchNorm': True},
'resnet_v1_50/block2/unit_2/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '6', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block2/unit_3/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Conv2D', 'id': '7', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/Conv2D', 'id': '7', 'place': 1, 'BatchNorm': True},
'resnet_v1_50/block2/unit_3/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '7', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block2/unit_4/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block2/unit_3/bottleneck_v1/conv2/Conv2D', 'id': '8', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/Conv2D', 'id': '8', 'place': 1, 'BatchNorm': True, 'split': 1},
'resnet_v1_50/block2/unit_4/bottleneck_v1/conv2/convolution':{'next': 'resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/Conv2D', 'id': '8', 'place': 1, 'BatchNorm': True, 'split': 2},
'resnet_v1_50/block2/unit_4/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '8', 'place': 2, 'BatchNorm': False},

'resnet_v1_50/block3/unit_1/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/Conv2D', 'id': '9', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block3/unit_1/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/Conv2D', 'id': '9', 'place': 1, 'BatchNorm': True},
'resnet_v1_50/block3/unit_1/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '9', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block3/unit_2/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/Conv2D', 'id': '10', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block3/unit_2/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/Conv2D', 'id': '10', 'place': 1, 'BatchNorm': True},
'resnet_v1_50/block3/unit_2/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '10', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block3/unit_3/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/Conv2D', 'id': '11', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/Conv2D', 'id': '11', 'place': 1, 'BatchNorm': True, 'split': 1},
'resnet_v1_50/block3/unit_3/bottleneck_v1/conv2/convolution':{'next': 'resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/Conv2D', 'id': '11', 'place': 1, 'BatchNorm': True, 'split': 2},
'resnet_v1_50/block3/unit_3/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '11', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block3/unit_4/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/Conv2D', 'id': '12', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/Conv2D', 'id': '12', 'place': 1, 'BatchNorm': True, 'split': 1},
'resnet_v1_50/block3/unit_4/bottleneck_v1/conv2/convolution':{'next': 'resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/Conv2D', 'id': '12', 'place': 1, 'BatchNorm': True, 'split': 2},
'resnet_v1_50/block3/unit_4/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '12', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block3/unit_5/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/Conv2D', 'id': '13', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/Conv2D', 'id': '13', 'place': 1, 'BatchNorm': True, 'split': 1},
'resnet_v1_50/block3/unit_5/bottleneck_v1/conv2/convolution':{'next': 'resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/Conv2D', 'id': '13', 'place': 1, 'BatchNorm': True, 'split': 2},
'resnet_v1_50/block3/unit_5/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '13', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block3/unit_6/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/Conv2D', 'id': '14', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/Conv2D':{'next': 'resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/Conv2D', 'id': '14', 'place': 1, 'BatchNorm': True, 'split': 1},
'resnet_v1_50/block3/unit_6/bottleneck_v1/conv2/convolution':{'next': 'resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/Conv2D', 'id': '14', 'place': 1, 'BatchNorm': True, 'split': 2},
'resnet_v1_50/block3/unit_6/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '14', 'place': 2, 'BatchNorm': False},

'resnet_v1_50/block4/unit_1/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/Conv2D', 'id': '15', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/convolution':{'next': 'resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/Conv2D', 'id': '15', 'place': 1, 'BatchNorm': True, 'split': 1},
'resnet_v1_50/block4/unit_1/bottleneck_v1/conv2/convolution_1':{'next': 'resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/Conv2D', 'id': '15', 'place': 1, 'BatchNorm': True, 'split': 2},
'resnet_v1_50/block4/unit_1/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '15', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block4/unit_2/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/Conv2D', 'id': '16', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/convolution':{'next': 'resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/Conv2D', 'id': '16', 'place': 1, 'BatchNorm': True, 'split': 1},
'resnet_v1_50/block4/unit_2/bottleneck_v1/conv2/convolution_1':{'next': 'resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/Conv2D', 'id': '16', 'place': 1, 'BatchNorm': True, 'split': 2},
'resnet_v1_50/block4/unit_2/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '16', 'place': 2, 'BatchNorm': False},
'resnet_v1_50/block4/unit_3/bottleneck_v1/conv1/Conv2D':{'next': 'resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/Conv2D', 'id': '17', 'place': 0, 'BatchNorm': True}, 
'resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/convolution':{'next': 'resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/Conv2D', 'id': '17', 'place': 1, 'BatchNorm': True, 'split': 1},
'resnet_v1_50/block4/unit_3/bottleneck_v1/conv2/convolution_1':{'next': 'resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/Conv2D', 'id': '17', 'place': 1, 'BatchNorm': True, 'split': 2},
'resnet_v1_50/block4/unit_3/bottleneck_v1/conv3/Conv2D':{'mask': True, 'id': '17', 'place': 2, 'BatchNorm': False},

'resnet_v1_50/conv256/Conv2D':{'next': 'resnet_v1_50/conv10/Conv2D', 'id': '18', 'BatchNorm': True},
'resnet_v1_50/conv71/Conv2D':{'next': 'resnet_v1_50/conv10/Conv2D', 'id': '22', 'BatchNorm': True},
'resnet_v1_50/conv81/Conv2D':{'next': 'resnet_v1_50/conv10/Conv2D', 'id': '26', 'BatchNorm': True},
'resnet_v1_50/conv91/Conv2D':{'next': 'resnet_v1_50/conv10/Conv2D', 'id': '30', 'BatchNorm': True},
'resnet_v1_50/conv70/Conv2D':{'next': 'resnet_v1_50/conv7/convolution', 'id': '19', 'BatchNorm': True},
'resnet_v1_50/conv80/Conv2D':{'next': 'resnet_v1_50/conv8/convolution', 'id': '23', 'BatchNorm': True},
'resnet_v1_50/conv90/Conv2D':{'next': 'resnet_v1_50/conv9/convolution', 'id': '27', 'BatchNorm': True},
'resnet_v1_50/conv7/convolution':{'next': 'resnet_v1_50/conv247/convolution', 'id': '20', 'BatchNorm': True},
'resnet_v1_50/conv8/convolution':{'next': 'resnet_v1_50/conv248/convolution', 'id': '24', 'BatchNorm': True},
'resnet_v1_50/conv9/convolution':{'next': 'resnet_v1_50/conv249/convolution', 'id': '28', 'BatchNorm': True},
'resnet_v1_50/conv247/convolution':{'next': 'resnet_v1_50/conv71/convolution', 'id': '21', 'BatchNorm': True},
'resnet_v1_50/conv248/convolution':{'next': 'resnet_v1_50/conv81/convolution', 'id': '25', 'BatchNorm': True},
'resnet_v1_50/conv249/convolution':{'next': 'resnet_v1_50/conv91/convolution', 'id': '29', 'BatchNorm': True},

'resnet_v1_50/conv10/Conv2D':{'next': 'resnet_v1_50/conv41/conv2d_transpose', 'id': '31', 'BatchNorm': True},
'resnet_v1_50/conv41/conv2d_transpose':{'next': 'resnet_v1_50/conv89/Conv2D', 'next_1': 'resnet_v1_50/conv912/Conv2D', 'id': '32', 'BatchNorm': True},
'resnet_v1_50/conv89/Conv2D':{'next': 'resnet_v1_50/conv96/Conv2D', 'id': '33', 'BatchNorm': True},

'resnet_v1_50/conv96/Conv2D':{'next': 'resnet_v1_50/conv16/conv2d_transpose', 'id': '34', 'BatchNorm': True},
'resnet_v1_50/conv16/conv2d_transpose':{'next': 'resnet_v1_50/conv88/Conv2D', 'next_1': 'resnet_v1_50/conv911/Conv2D', 'id': '35', 'BatchNorm': True},
'resnet_v1_50/conv88/Conv2D':{'next': 'resnet_v1_50/conv95/Conv2D', 'id': '36', 'BatchNorm': True},
'resnet_v1_50/conv95/Conv2D':{'next': 'resnet_v1_50/conv78/Conv2D', 'id': '37', 'BatchNorm': True},  
}