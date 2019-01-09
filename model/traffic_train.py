#!/usr/bin/env python
# encoding: utf-8
import matplotlib.pylab as plt
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
sys.path.append("../process")#添加其他文件夹
import data_input#导入其他模块
from traffic_network import Lenet

def train(aug, model,train_x,train_y,test_x,test_y):

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",metrics=["accuracy"])#配置
    #model.fit(train_x,train_y,batch_size,epochs,validation_data=(test_x,test_y))
    _history = model.fit_generator(aug.flow(train_x,train_y,batch_size=batch_size),
                        validation_data=(test_x,test_y),steps_per_epoch=len(train_x)//batch_size,
                        epochs=epochs,verbose=1)
    #拟合，具体fit_generator请查阅其他文档,steps_per_epoch是每次迭代，需要迭代多少个batch_size，validation_data为test数据，直接做验证，不参与训练

    plt.style.use("ggplot")#matplotlib的美化样式
    plt.figure()
    N = epochs
    plt.plot(np.arange(0,N),_history.history["loss"],label ="train_loss")#model的history有四个属性，loss,val_loss,acc,val_acc
    plt.plot(np.arange(0,N),_history.history["val_loss"],label="val_loss")
    plt.plot(np.arange(0,N),_history.history["acc"],label="train_acc")
    plt.plot(np.arange(0,N),_history.history["val_acc"],label="val_acc")
    plt.title("loss and accuracy")
    plt.xlabel("epoch")
    plt.ylabel("loss/acc")
    plt.legend(loc="best")
    plt.savefig("../result/result.png")
    plt.show()

if __name__ =="__main__":
    channel = 3
    height = 32
    width = 32
    class_num = 62
    norm_size = 32#参数
    batch_size = 32
    epochs = 40
    model = Lenet.neural(channel=channel, height=height,
                         width=width, classes=class_num)#网络
    train_x, train_y = data_input.load_data("../data/train", norm_size, class_num)
    test_x, test_y = data_input.load_data("../data/test", norm_size, class_num)#生成数据

    aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
                       height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
                       horizontal_flip=True,fill_mode="nearest")#数据增强，生成迭代器

    train(aug,model,train_x,train_y,test_x,test_y)#训练





