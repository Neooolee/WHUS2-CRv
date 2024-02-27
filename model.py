# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:56:44 2018

@author: lijun
"""
import tensorflow as tf
from allnet import *
def CR4S2(input_dim, gf_dim=64,training=False, name="CR4S2"):
    dropout_rate = 0.2
    input_ = tf.keras.layers.Input(shape=[None,None,input_dim[0]]) 
    input_1 = tf.keras.layers.Input(shape=[None,None,input_dim[1]])
    input_2 = tf.keras.layers.Input(shape=[None,None,input_dim[2]])
    e10 = lrelu(conv2d(input_,gf_dim,kernel_size=3,stride=1))
    e1 = lrelu(instance_norm(DDSC(e10,gf_dim,stride=2),training))
    e1 = max_pooling(e10,2,2)+e1
        # e1 is (128 x 128 x self.gf_dim)

    e20 = lrelu(conv2d(input_1,gf_dim,kernel_size=3,stride=1))        
    c12 = tf.concat([e1,e20],axis=-1)
    e2 = lrelu(instance_norm(DDSC(c12,gf_dim,kernel_size=4,stride=3),training))
    e2 = max_pooling(e1,3,3)+max_pooling(e20,3,3)+e2
        # e2 is (64 x 64 x self.gf_dim*2)
    # p2=max_pooling(e2,3,3)
    e30 = lrelu(conv2d(input_2,gf_dim,kernel_size=3,stride=1))
    c123 = tf.concat([e2,e30],axis=-1)               
    e3 = lrelu(instance_norm(DDSC(c123,gf_dim,stride=2),training))
    e3 = max_pooling(e2,2,2)+max_pooling(e30,2,2)+e3
        # e3 is (32 x 32 x self.gf_dim*4)
    # p3 = max_pooling(e3,2,2)
    e4 = lrelu(instance_norm(DDSC(e3,gf_dim,stride=2),training))
        # e3 is (32 x 32 x self.gf_dim*4)
    e4= max_pooling(e3,2,2)+e4
    

    r1=SDRB(e4,3,1,2,training,norm='instance_norm',activ="lrelu")
    r2=SDRB(r1,3,1,2,training,norm='instance_norm',activ="lrelu")
    r3=SDRB(r2,3,1,3,training,norm='instance_norm',activ="lrelu")
    r4=SDRB(r3,3,1,3,training,norm='instance_norm',activ="lrelu")
    r5=SDRB(r4,3,1,4,training,norm='instance_norm',activ="lrelu")
    r6=SDRB(r5,3,1,4,training,norm='instance_norm',activ="lrelu")

    c0 = tf.concat([e4,r2,r4,r6], axis=-1)
    c0 = relu(instance_norm(conv2d(c0, gf_dim*2, stride=1),training))

    d1 = relu(instance_norm(deconv2d(c0, gf_dim,stride=2)))
    # d1 = dropout(d1, dropout_rate,training=training)
    d1 = tf.concat([d1, e3], 3)
    #d1 = relu(instance_norm(DDSC(d1,gf_dim,stride=1),training))
    d2 = relu(instance_norm(deconv2d(d1, gf_dim,stride=2)))
    d2 = tf.concat([d2, c123], 3)
    #d2 = relu(instance_norm(DDSC(d2,gf_dim,stride=1),training))
    output2 = tf.math.sigmoid(conv2d(d2,input_dim[2]-1,stride=1))

    d3 = relu(instance_norm(deconv2d(d2, gf_dim,stride=3)))
    d3 = tf.concat([d3, c12],3)
    #d3 = relu(instance_norm(DDSC(d3,gf_dim,stride=1),training))          
    output1 = tf.math.sigmoid(conv2d(d3,input_dim[1],stride=1))

    d4 = relu(instance_norm(deconv2d(d3, gf_dim,stride=2)))
    d4 = tf.concat([d4, e10],3) 
    #d4 = relu(instance_norm(DDSC(d4,gf_dim,stride=1),training))         
    output = tf.math.sigmoid(conv2d(d4,input_dim[0],stride=1))

    return tf.keras.Model([input_,input_1,input_2],[output,output1,output2],name=name)
