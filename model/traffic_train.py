#!/usr/bin/env python
# encoding: utf-8
import matplotlib.pylab as plt
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys
sys.path.append("../process")
import data_input
from traffic_network import Lenet

def train(aug, model,train_x,train_y,test_x,test_y):

    model.compile(loss="categorical_crossentropy",
                  optimizer="Adam",metrics=["accuracy"])
    #model.fit(train_x,train_y,batch_size,epochs,validation_data=(test_x,test_y))
    _history = model.fit_generator(aug.flow(train_x,train_y,batch_size=batch_size),
                        validation_data=(test_x,test_y),steps_per_epoch=len(train_x)//batch_size,
                        epochs=epochs,verbose=1)

    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0,N),_history.history["loss"],label ="train_loss")
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
    norm_size = 32
    batch_size = 32
    epochs = 40
    model = Lenet.neural(channel=channel, height=height,
                         width=width, classes=class_num)
    train_x, train_y = data_input.load_data("../data/train", norm_size, class_num)
    test_x, test_y = data_input.load_data("../data/test", norm_size, class_num)

    aug = ImageDataGenerator(rotation_range=30,width_shift_range=0.1,
                       height_shift_range=0.1,shear_range=0.2,zoom_range=0.2,
                       horizontal_flip=True,fill_mode="nearest")

    train(aug,model,train_x,train_y,test_x,test_y)





