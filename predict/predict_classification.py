#!/usr/bin/env python
# encoding: utf-8
import keras
from keras.preprocessing.image import img_to_array
import imutils.paths as paths
import cv2
import os
import numpy as np
import matplotlib.pylab as plt

#import sys
#sys.path.append("../process")
#import data_input

channel = 3
height = 32
width = 32
class_num = 62
norm_size = 32  # 参数
batch_size = 32
epochs = 40

test_path = "../data/test"
image_paths = sorted(list(paths.list_images(test_path)))#提取所有图片地址
model = keras.models.load_model("traffic_model.h5")#加载模型
for each in image_paths:
    image = cv2.imread(each)
    image = cv2.resize(image,(norm_size,norm_size))
    image = img_to_array(image)/255.0
    result = model.predict(image)#分类预测
    proba = np.max(result)#提取最大概率
    predict_label = np.argmax(result)#提取最大概率下标
    label = int(each.split(os.path.sep)[-2])#提取标签
    plt.imshow(image[0])#显示图片
    plt.title("label:{},predict_label:{}, proba:{:.2f}".format(label,predict_label,proba))
    plt.show()





#test_x,test_y = data_input.load_data("../data/train",norm_size,class_num)
#model = keras.models.load_model("traffic_model.h5")
#result = model.predict( ,batch_size=1,verbose=0)
#print(result)
#print(result.shape)
