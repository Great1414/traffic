# 一、项目简介

练习搭建CNN，做交通标志图片识别
参加文献：https://www.cnblogs.com/skyfsm/p/8051705.html
---
## (一)、库安装
基于TensorFlow后端的keras
---
## (二)、结构说明
本项目的文件主要分为3部分：
 * 1.data-输入数据
test:测试数据（不全）
train：训练数据（不全）
数据源：https://pan.baidu.com/s/1o8uZ9k2

 * 2.model-网络搭建及训练
traffic_network.py搭建LeNet网络
traffic_train.py训练网络

 * 3.process-数据处理
data_input.py读取数据，x和y的创建，乱序，归一化，one_hot等

 * 4.result-预测结果

二、测试
=========

```python
cd clone model#文件目录
执行traffic_train.py文件
