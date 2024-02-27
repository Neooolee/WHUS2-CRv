# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:07:54 2018

@author: Neoooli
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from collections import OrderedDict
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras.models import Sequential
from gdaldiy import *
import glob
import os
kernel_regularizers=regularizers.l2(1e-4)
def conv2d(input_,output_dim,kernel_size=3,stride=2,padding="SAME",biased=True):
    return Conv2D(output_dim,kernel_size=[kernel_size,kernel_size],strides=[stride,stride],padding=padding,use_bias=biased,
            kernel_regularizer=kernel_regularizers)(input_)
def deconv2d(input_,output_dim,kernel_size=4,stride=2,padding="SAME",biased=True):
    return Conv2DTranspose(output_dim,kernel_size=[kernel_size,kernel_size],strides=[stride,stride],padding=padding,use_bias=biased,
            kernel_regularizer=kernel_regularizers)(input_)
def DSC(input_,output_dim,kernel_size=3, stride=1, padding="SAME",scale=1, biased=True):
    return SeparableConv2D(output_dim,kernel_size=[kernel_size,kernel_size],strides=[stride,stride],padding=padding,depth_multiplier=scale,use_bias=biased,
    depthwise_regularizer=kernel_regularizers,pointwise_regularizer=kernel_regularizers)(input_)
def DDSC(input_,output_dim,kernel_size=3,stride=1, padding="SAME",scale=1,biased=True):
    depth_output1=DepthwiseConv2D(kernel_size=kernel_size,strides=stride,padding=padding,depth_multiplier=scale,use_bias=biased,
                        depthwise_regularizer=kernel_regularizers)(input_)
    depth_output20=DepthwiseConv2D(kernel_size=kernel_size,strides=stride,padding=padding,depth_multiplier=scale,use_bias=biased,
                        depthwise_regularizer=kernel_regularizers)(input_)
    depth_output21=DepthwiseConv2D(kernel_size=3,strides=1,padding=padding,depth_multiplier=scale,use_bias=biased,
                        depthwise_regularizer=kernel_regularizers)(depth_output20)                 

    output = tf.concat([depth_output1,depth_output21],axis=-1)
    output = conv2d(output,output_dim,kernel_size=1,stride=1,padding=padding,biased=biased)
    return output
def MDSC(input_,output_dim,kernel_list=[3,5], stride=1, padding="SAME",scale=1,biased=True):
    output_list=[]
    for i in range(len(kernel_list)):
        depth_output=DepthwiseConv2D(kernel_size=kernel_list[i],strides=stride,padding=padding,depth_multiplier=scale,use_bias=biased,
                        depthwise_regularizer=kernel_regularizers)(input_)
        output_list.append(depth_output)
    output = tf.concat(output_list,axis=-1)
    output = conv2d(output,output_dim,kernel_size=1,stride=1,padding=padding,biased=biased)
    return output
def SDSC(input_,output_dim,kernel_list=[3,5], stride=1, padding="SAME",scale=1,biased=True):
    output_list=[]
    for i in range(len(kernel_list)):
        depth_output=DepthwiseConv2D(kernel_size=kernel_list[i],strides=stride,padding=padding,depth_multiplier=scale,use_bias=biased,
                        depthwise_regularizer=kernel_regularizers)(input_)
        output_list.append(depth_output)
    output = tf.concat(output_list,axis=-1)
    output = conv2d(output,output_dim,kernel_size=1,stride=1,padding=padding,biased=biased)
    return output
def SDC(input_,output_dim, kernel_size=3,stride=1,dilation=2,padding='SAME', biased=True):
    """
    Smoothed dilated conv2d via the Separable and Shared Convolution (SSC) without BN or relu.
    """
    input_dim = input_.shape[-1]
    fix_w_size = dilation * 2 - 1
    eo = tf.expand_dims(input_,-1)
    o = Conv3D(1,kernel_size=[fix_w_size, fix_w_size,1],strides=[stride,stride,stride],padding=padding,use_bias=biased,
        kernel_regularizer=kernel_regularizers)(eo)
    o = eo + o
    o = tf.squeeze(o,-1)
    o = Conv2D(output_dim,kernel_size=[kernel_size,kernel_size],strides=[stride,stride],padding=padding,dilation_rate=(dilation, dilation),use_bias=biased,
        kernel_regularizer=kernel_regularizers)(o)
    return o
def SDRB(input_, kernel_size=3,stride=1,dilation=2,training=False,biased=True,norm='instance_norm',activ="lrelu"):
    output_dim=input_.get_shape()[-1]
    sconv1=SDC(input_,output_dim, kernel_size,stride,dilation,biased=biased)
    sconv1=norm_layer(sconv1,'instance_norm',training)
    sconv1= act(sconv1,activ)
    sconv2=SDC(sconv1,output_dim, kernel_size,stride,dilation,biased=biased)
    sconv2=norm_layer(sconv2,'instance_norm',training)
    return act(sconv2+input_,activ) 
def relu(input_):
    return ReLU()(input_)
def lrelu(input_):
    return LeakyReLU()(input_)
def avg_pooling(input_,kernel_size=2,stride=2,padding="same"):
    return tf.keras.layers.AveragePooling2D((kernel_size,kernel_size),stride,padding)(input_)
def max_pooling(input_,kernel_size=2,stride=2,padding="same"):
    return tf.keras.layers.MaxPool2D((kernel_size,kernel_size),stride,padding)(input_)
def dropout(input_,rate=0.2,training=True):
    """
    rate是丢掉多少神经元.
    """  
    return tf.keras.layers.Dropout(rate)(input_,training)
def GAP(input_):
    return GlobalAveragePooling2D()(input_)
def batch_norm(input_,training=True):
    return BatchNormalization()(input_,training)
class InstanceNormalization(tf.keras.layers.Layer):
  def __init__(self, epsilon=1e-5):
    super(InstanceNormalization, self).__init__()
    self.epsilon = epsilon

  def build(self, input_shape):
    self.scale = self.add_weight(
        name='scale',
        shape=input_shape[-1:],
        initializer=tf.random_normal_initializer(1., 0.02),
        trainable=True)

    self.offset = self.add_weight(
        name='offset',
        shape=input_shape[-1:],
        initializer='zeros',
        trainable=True)

  def call(self, x):
    mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
    inv = tf.math.rsqrt(variance + self.epsilon)
    normalized = (x - mean) * inv
    return self.scale * normalized + self.offset

def instance_norm(input_,training=True):
    return InstanceNormalization()(input_)

def norm_layer(input_,norm_type='batch_norm',training=True):
    if norm_type==None:
        return input_
    elif norm_type=='batch_norm':
        return BatchNormalization()(input_,training)
    elif norm_type=='instance_norm':
        return InstanceNormalization()(input_)

def act(input_,activation='relu'):
    if activation==None:
        return input_
    elif activation=='relu':
        return ReLU()(input_)
    elif activation=='lrelu':
        return LeakyReLU(alpha=0.2)(input_)

def CBRBlock(input_,output_dim=64,kernel_size=3,stride=2,norm_type="batch_norm",activation="relu",training=True):
    """
    Convolution with/out BN/IN or relu/lrelu follow.
    """  
    x=conv2d(input_,output_dim,kernel_size,stride)
    x=norm_layer(x,norm_type,training)
    x=act(x,activation)
    return x
def BRCBlock(input_,output_dim=64,kernel_size=3,stride=2,norm_type="batch_norm",activation="relu",training=True):
    """
    Convolution with/out BN/IN or relu/lrelu follow.
    """      
    x=norm_layer(input_,norm_type,training)
    x=act(x,activation)
    x=conv2d(x,output_dim,kernel_size,stride)
    return x
def DeConvBlock(input_,output_dim=64,kernel_size=4,stride=2,norm_type=None,activation=None,training=True):
    """
    DeConvolution with/out BN/IN or relu/lrelu follow.
    """   
    x=deconv2d(input_,output_dim,kernel_size,stride)
    x=norm_layer(x,norm_type,training)
    x=act(x,activation)
    return x 
def res_block(input_, output_dim, kernel_size = 3, stride = 1):
    conv2dc0 = conv2d(input_, output_dim = output_dim, kernel_size = kernel_size, stride = stride)
    conv2dc0_norm = instance_norm(conv2dc0)
    conv2dc0_relu = relu(conv2dc0_norm)
    conv2dc1 = conv2d(conv2dc0_relu, output_dim = output_dim, kernel_size = kernel_size, stride = stride)
    conv2dc1_norm = instance_norm(conv2dc1)
    add_raw = input_ + conv2dc1_norm
    output = relu(add_raw)
    return output   
def diydecay(steps,baselr,cycle_step=100000,decay_steps=100,decay_rate=0.98):
    n=steps//cycle_step
    clr=baselr*(0.96**n)   
    steps=steps-n*cycle_step
    k=steps//decay_steps
    dlr = clr*(decay_rate**k)      
    return dlr
def decay(global_steps,baselr,start_decay_step=100000,cycle_step=100000,decay_steps=100,decay_rate=0.98):
    lr=np.where(np.greater_equal(global_steps,start_decay_step),
                diydecay(global_steps-start_decay_step,baselr,cycle_step,decay_steps,decay_rate),
                baselr)
    return lr
def make_train_data_list(data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filepath= glob.glob(os.path.join(data_path, "*")) #读取全部的x域图像路径名称列表
    image_path_lists=[]
    for i in range(len(filepath)):
         path=glob.glob(os.path.join(filepath[i], "*"))
         for j in range(len(path)):
             image_path_lists.append(path[j]) #将x域图像数量与y域图像数量对齐
    return image_path_lists
    
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(src-dst))
def l2_loss(x):
    return tf.sqrt(tf.reduce_sum(x**2))
def gan_loss(src, dst): #定义gan_loss，在这里用了二范数
    return tf.reduce_mean((src- dst)**2)
def liner_2(input_):#2%线性拉伸,返回0~1之间的值
    def strech(img):
        low,high=np.percentile(img,(2,98))
        img[low>img]=low
        img[img>high]=high
        return (img-low)/(high-low+1e-10)
    if len(input_.shape)>2:
        for i in range(input_.shape[-1]):
            input_[:,:,i]=strech(input_[:,:,i])
    else:
        input_=strech(input_)    
    return input_

def get_write_picture(row_list): #get_write_picture函数得到训练过程中的可视化结果
    row_=[]    
    for i in range(len(row_list)):
        row=row_list[i] 
        col_=[]
        for image in row:
            x_image=image[:,:,[2,1,0]]
            if i<1:
                x_image=liner_2(x_image)
            col_.append(x_image)
        row_.append(np.concatenate(col_,axis=1))
    if len(row_list)==1:
        output = np.concatenate(col_,axis=1)
    else:
        output = np.concatenate(row_, axis=0) #得到训练中可视化结果
    return output*255 
def l2_loss(x):
    return tf.sqrt(tf.reduce_sum(x**2))
def grad(src):
    g_src_x = src[:, 1:, :, :] - src[:, :-1, :, :]
    g_src_y = src[:, :, 1:, :] - src[:, :, :-1, :]
    return g_src_x,g_src_y

def gradxy(src):
    src = tf.pad(src,[[0,0],[1,0],[1,0],[0,0]],mode="SYMMETRIC")
    I_x = src[:,:,1:,1:]-src[:,:,:-1,1:]
    I_y = src[:,:,1:,1:]-src[:,:,1:,:-1]
    return I_x,I_y
def grad_map(src):
    I_x,I_y = gradxy(src)
    return tf.math.sqrt(tf.math.square(I_x)+tf.math.square(I_y)+1e-6)       

  
def randomflip(input_,n):
    #生成-3到2的随机整数，-1顺时针90度，-2顺时针180，-3顺时针270,0垂直翻转，1水平翻转，2不变
    if n<0:
        return np.rot90(input_,n)
    elif -1<n<2:
        return np.flip(input_,n)
    else: 
        return input_
def read_img(datapath,scale=255):
    img=imgread(datapath)
    img[img>scale]=scale
    img=img/scale   
    return img
def read_imgs(datapath,scale=255,k=2):
    img_list=[]
    l=len(datapath)
    for i in range(l):
        img=read_img(datapath[i],scale)
        img = randomflip(img,k)
        img=img[np.newaxis,:]
        img_list.append(img)    
    imgs=np.concatenate(img_list,axis=0)
    return tf.convert_to_tensor(imgs,tf.float32)

def iterate_img(file_list,batch_size=1,rn_list=None,scale=10000,num__cores=tf.data.experimental.AUTOTUNE):
    if np.any(rn_list)==None:
        rn_list= [2 for _ in range(len(file_list))]#如果不指定翻转参数，就不翻转
    def gen(file_name,k):
        # print('load img')
        img=read_img(file_name.numpy().decode('utf-8'),scale)
        img=randomflip(img,k)
        return img
    def iterator():       
        dataset=tf.data.Dataset.from_tensor_slices((file_list,rn_list))
        dataset=dataset.map(lambda x,y: tf.py_function(gen,[x,y],tf.float32),num_parallel_calls=num__cores)
        dataset=dataset.batch(batch_size)
        dataset=dataset.prefetch(num__cores)
        return dataset
    return iterator()

