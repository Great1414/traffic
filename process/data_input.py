#!/usr/bin/env python
# encoding: utf-8
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import imutils
import cv2
import numpy as np
import random
import sys
import os

def load_data(path,norm_size,class_num):
    data = []
    label = []
    image_paths = sorted(list(paths.list_images(path)))
    random.seed(0)
    random.shuffle(image_paths)
    for each_path in image_paths:
        image = cv2.imread(each_path)
        image = cv2.resize(image,(norm_size,norm_size))
        image = img_to_array(image)
        data.append(image)
        print(each_path)
        maker = int(each_path.split(os.path.sep)[-2])
        print(maker)

        label.append(maker)
    data = np.array(data,dtype="float")/255.0
    label = np.array(label)
    label = to_categorical(label,num_classes=class_num)
    return data,label
