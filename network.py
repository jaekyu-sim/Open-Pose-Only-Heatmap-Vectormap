
# coding: utf-8

# In[3]:


import tensorflow as tf
import numpy as np

def block_vgg_19(input_image):
    #input => 228x228x3
    #block1
    net = tf.layers.conv2d(inputs=input_image, filters=64, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#228x228x64
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#228x228x64
    net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2))#114x114x64

    #block2
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#114x114x128
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#114x114x128
    net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2))#57x57x128

    #block3
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#57x57x256
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#57x57x256
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#57x57x256
    net = tf.layers.conv2d(inputs=net, filters=256, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#57x57x256
    net = tf.layers.max_pooling2d(inputs=net, pool_size=(2, 2), strides=(2, 2))#28x28x256

    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#28x28x512
    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(3, 3), padding="SAME", strides=(1,1))
    net = tf.nn.relu(net)
    #print(np.shape(net))#28x28x512

    return net

def block_stage_1_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 34
    return net

def block_stage_2_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 34
    return net
def block_stage_3_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 34
    return net
def block_stage_4_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 34
    return net
def block_stage_5_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 34
    return net
def block_stage_6_branch1(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=34, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 34
    return net

def block_stage_1_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(3, 3), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=512, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 17
    return net

def block_stage_2_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 17
    return net
def block_stage_3_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 17
    return net
def block_stage_4_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 17
    return net
def block_stage_5_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 17
    return net
def block_stage_6_branch2(feature):
    net = tf.layers.conv2d(inputs=feature, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(7, 7), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=128, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=17, kernel_size=(1, 1), padding="SAME", strides=(1, 1))
    #net -> 29, 29, 17
    return net

