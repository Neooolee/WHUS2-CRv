# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:08:04 2018

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
from gdaldiy import *
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--x_test_data_path", default='f:/lijun/data/graduatedata/cloudremoval/S2A/WHUS2-CR/test/cloudDNclips/10m', help="path of x test datas.") #x域的测试图片路径
parser.add_argument("--image_size", type=int, default=[384,192,64], help="load image size") #网络输入的尺度
parser.add_argument("--bands", type=int, default=[4,6,3], help="load batch size") #batch_size
parser.add_argument("--batch_size", type=int, default=1, help="load batch size")
parser.add_argument("--snapshots", default='./snapshots/',help="Path of Snapshots") #读取训练好的模型参数的路径
parser.add_argument("--out_dir", default='./test_out/',help="Output Folder") #保存x域的输入图片与生成的y域图片的路径
args = parser.parse_args()

def make_test_data_list(data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filepath= glob.glob(os.path.join(data_path, "*")) #读取全部的x域图像路径名称列表
    image_path_lists=[]
    for i in range(len(filepath)):
         path=glob.glob(os.path.join(filepath[i], "*"))
         for j in range(len(path)):
             image_path_lists.append(path[j]) 
    return image_path_lists

def main(num):
    if not os.path.exists(args.out_dir): #如果保存x域测试结果的文件夹不存在则创建
        os.makedirs(args.out_dir)      
    x_datalists= make_test_data_list(args.x_test_data_path) #得到待测试的x域和y域图像路径名称列表
    model = CR4S2(args.bands,training=False,name=CR4S2")
    
    ckpt=tf.train.Checkpoint(G_Net=model)
    
    modelname=args.snapshots+'ckpt-'+num
    ckpt.restore(modelname)

    print('开始处理',datetime.now())
    starttime=datetime.now()  
    for i in range(len(x_datalists)):
        out_path=x_datalists[i].split('\\')
        testx = read_imgs(x_datalists[i:i+1],10000)
        x1list=[i.replace('10m','20m') for i in x_datalists[i:i+1]]
        x2list=[i.replace('10m','60m') for i in x_datalists[i:i+1]]
        testx1 = read_imgs(x1list,10000) 
        testx2 = read_imgs(x2list,10000)  
        out_list=model([testx,testx1,testx2])    
        write_image=out_list[0].numpy()[0,:,:,:]*10000
        write_image1=out_list[1].numpy()[0,:,:,:]*10000
        write_image2=out_list[2].numpy()[0,:,:,:]*10000
        savepath=args.out_dir+'10m/'+out_path[-2]
        savepath1=args.out_dir+'20m/'+out_path[-2]
        savepath2=args.out_dir+'60m/'+out_path[-2]
        if not os.path.exists(savepath): #如果保存x域测试结果的文件夹不存在则创建
            os.makedirs(savepath)
        if not os.path.exists(savepath1): #如果保存x域测试结果的文件夹不存在则创建
            os.makedirs(savepath1)
        if not os.path.exists(savepath2): #如果保存x域测试结果的文件夹不存在则创建
            os.makedirs(savepath2)
        savepath=savepath+'/'+out_path[-1].split('.')[-2]+'.tif'
        imgwrite(savepath,write_image)
        savepath1=savepath1+'/'+out_path[-1].split('.')[-2]+'.tif'
        imgwrite(savepath1,write_image1)
        savepath2=savepath2+'/'+out_path[-1].split('.')[-2]+'.tif'
        imgwrite(savepath2,write_image2)
        sys.stdout.write('\r'+'已处理：{:.0%}'.format(i/len(x_datalists)))
        sys.stdout.flush()
    endtime=datetime.now()     
    print('结束时间：',endtime)
    deltatime=endtime-starttime
    print('检测用时:',deltatime) 
    print('平均用时',deltatime)
        

if __name__ == '__main__':
    ckptnum=10000
    main(str(ckptnum))
    compute_ps(ckptnum)

