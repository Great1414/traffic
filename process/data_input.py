#!/usr/bin/env python
# encoding: utf-8
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import cv2
import numpy as np
import random
import os

def load_data(path,norm_size,class_num):
    data = []
    label = []
    image_paths = sorted(list(paths.list_images(path)))#imutils模块中paths可以读取文件路径
    random.seed(0)
    random.shuffle(image_paths)#将所有的文件路径打乱
    for each_path in image_paths:
        image = cv2.imread(each_path)#读取文件
        image = cv2.resize(image,(norm_size,norm_size))#统一图片尺寸
        image = img_to_array(image)
        data.append(image)
        maker = int(each_path.split(os.path.sep)[-2])#切分文件目录，类别为文件夹整数变化，从0-61.如train文件下00014，label=14
        label.append(maker)
    data = np.array(data,dtype="float")/255.0#归一化
    label = np.array(label)
    label = to_categorical(label,num_classes=class_num)#one-hot
    return data,label
