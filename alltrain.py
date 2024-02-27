# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:57:36 2018

@author: Neoooli
"""

from __future__ import print_function
 
import argparse
from datetime import datetime
from random import shuffle
import random
import os
import sys
import time
import math
import tensorflow as tf

import numpy as np
import glob
from PIL import Image
from model import *
from allnet import *
# from utils import *
from gdaldiy import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='')
parser.add_argument("--snapshot_dir", default='./snapshots/', help="path of snapshots") #保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=[384,192,64], help="load image size") #网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam') #基础学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch') #训练的epoch数量
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda") #训练中L1_Loss前的乘数
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument('--beta2', dest='beta2', type=float, default=0.9, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=100, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=1000, help="times to write.") #训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=10000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--x_train_data_path", default=r'F:\WHU\WHUS2-CR\WHUS2-CRv\train/clearDNclips/10m', help="path of x training datas.") #x域的训练图片路径
parser.add_argument("--y_train_data_path", default=r'F:\WHU\WHUS2-CR\WHUS2-CRv\train/clearDNclips/10m', help="path of y training datas.") #y域的训练图片路径
# parser.add_argument("--z_train_data_path", default='E:/lijun/data/allcloud/', help="path of z training datas.") #y域的训练图片路径
parser.add_argument("--batch_size", type=int, default=1,help="load batch size") #batch_size
parser.add_argument("--bands", type=int, default=[4,6,3], help="load batch size") #batch_size
parser.add_argument("--classes", type=int, default=1, help="load batch size")
parser.add_argument("--output_level", type=int, default=1, help="load batch size")
args = parser.parse_args()

class maintrain(object):
    """docstring for maintrain"""
    def __init__(self):
        super(maintrain, self).__init__() 
        self.G_Net = CR4S2(args.bands,training=True,name="CR4S2")
        self.g_optimizer = tf.keras.optimizers.Adam(args.base_lr,args.beta1,args.beta2)
        self.ckpt = tf.train.Checkpoint(G_Net=self.G_Net)

    @tf.function                
    def train_step(self,ximage_list,yimage_list,lr):
        self.g_optimizer.lr.assign(lr)

        with tf.GradientTape(persistent=True) as tape:
            fake_y_list=self.G_Net(ximage_list)
            indentity_y_list=self.G_Net(yimage_list)
            g_loss=0

            for i in range(len(fake_y_list)-1):
                g_loss+=args.lamda*l1_loss(fake_y_list[i],yimage_list[i])\
                    +l1_loss(grad_map(fake_y_list[i]),grad_map(yimage_list[i]))\
                    +0.1*l1_loss(indentity_y_list[i],yimage_list[i])\
                    +0.01*l1_loss(grad_map(indentity_y_list[i]),grad_map(yimage_list[i]))
            g_loss+=args.lamda*l1_loss(fake_y_list[2],yimage_list[2][:,:,:,0:2])\
                    +l1_loss(grad_map(fake_y_list[2]),grad_map(yimage_list[2][:,:,:,0:2]))\
                    +0.1*l1_loss(indentity_y_list[2],yimage_list[2][:,:,:,0:2])\
                    +0.01*l1_loss(grad_map(indentity_y_list[2]),grad_map(yimage_list[2][:,:,:,0:2]))  
        grads_g=tape.gradient(g_loss,self.G_Net.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads_g,self.G_Net.trainable_variables))

        return g_loss,fake_y_list[0]

    def train(self,x_datalists,y_datalists):
        print ('Start Training')
        #存储训练日志
        train_summary_writer = tf.summary.create_file_writer(args.snapshot_dir)
        ckpt_manager = tf.train.CheckpointManager(self.ckpt,args.snapshot_dir, max_to_keep=100)
        if ckpt_manager.latest_checkpoint:
            self.ckpt.restore(ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')
            path=ckpt_manager.latest_checkpoint
            step=int(path.split('-')[-1])
        else: 
            step=1       
        leny=len(y_datalists)
        start_epoch=(step*args.batch_size)//leny+1
        scale=10000
        for epoch in range(start_epoch,args.epoch): #训练epoch数       
               #每训练一个epoch，就打乱一下x域图像顺序
            shuffle(y_datalists) #每训练一个epoch，就打乱一下y域图像顺序  
            y_datalists1= [name.replace('10m','20m') for name in y_datalists]
            y_datalists2= [name.replace('10m','60m') for name in y_datalists]

            x_datalists= [name.replace('clear','cloud') for name in y_datalists]
            x_datalists1= [name.replace('clear','cloud') for name in y_datalists1]
            x_datalists2= [name.replace('clear','cloud') for name in y_datalists2]

            k_list = np.random.randint(low=-3, high=3,size=leny)
            # ey_datalists=expand_list(y_datalists,lenx)
            x_dataset=iterate_img(x_datalists,args.batch_size,k_list,scale)
            x_dataset1=iterate_img(x_datalists1,args.batch_size,k_list,scale)
            x_dataset2=iterate_img(x_datalists2,args.batch_size,k_list,scale)
            y_dataset=iterate_img(y_datalists,args.batch_size,k_list,scale)
            y_dataset1=iterate_img(y_datalists1,args.batch_size,k_list,scale)
            y_dataset2=iterate_img(y_datalists2,args.batch_size,k_list,scale)

            for batch_inputx_img,batch_inputx_img1,batch_inputx_img2,batch_inputy_img,batch_inputy_img1,batch_inputy_img2\
                in tf.data.Dataset.zip((x_dataset,x_dataset1,x_dataset2,y_dataset,y_dataset1,y_dataset2)):

                lr=tf.convert_to_tensor(decay(step,args.base_lr,start_decay_step=100000,cycle_step=100000,decay_steps=100),tf.float32)             
                gl,fake_y= self.train_step([batch_inputx_img,batch_inputx_img1,batch_inputx_img2],
                            [batch_inputy_img,batch_inputy_img1,batch_inputy_img2],
                            lr) #得到每个step中的生成器和判别器loss
                step=step+1       
                if step% args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
                    with train_summary_writer.as_default():
                        tf.summary.scalar('g_loss',gl.numpy(),step)
                        tf.summary.scalar('lr',lr.numpy(),step)
                if step% args.save_pred_every == 0: #每过summary_pred_every次保存训练日志
                    ckpt_manager.save(checkpoint_number=step)
                if step % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果
                    write_image = get_write_picture([[batch_inputx_img[0].numpy(),
                            batch_inputy_img[0].numpy(),
                            fake_y[0].numpy()]]) #得到训练的可视化结果
                    write_image_name = args.out_dir + "/out"+ str(epoch)+'_'+str(step)+ ".png" #待保存的训练可视化结果路径与名称
                    imgwrite(write_image_name,np.uint8(write_image)) #保存训练的可视化结果
                    print('epoch step     a_loss      lr')
                    print('{:d}     {:d}    {:.3f}      {:.8f} '.format(epoch,step,gl.numpy(),lr.numpy()))
                if step==1000000:
                    exit()
            ckpt_manager.save(checkpoint_number=epoch)
            # if epoch==40:
                # ckpt_manager.save(checkpoint_number=epoch)
                # exit()

def main():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')#获取GPU列表
    print(gpus)
    tf.config.experimental.set_memory_growth(gpus[0], True)#设置GPU动态申请
    ##限制消耗固定大小的显存（程序不会超出限定的显存大小，若超报错）
    # tf.config.experimental.set_virtual_device_configuration(
    # gpus[0],
    # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000),
    # tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)
    x_datalists = make_train_data_list(args.x_train_data_path) #得到数量相同的x域和y域图像路径名称列表
    y_datalists = make_train_data_list(args.y_train_data_path)
    
    maintrain_object=maintrain()
    maintrain_object.train(x_datalists,y_datalists)
                
if __name__ == '__main__':
    main()

