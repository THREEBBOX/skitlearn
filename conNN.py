import tensorflow as tf
"""
l1_regularizer
为各层添加L1、L2正则化方法
"""
x=tf.placeholder(tf.float32,shape=(None,28,25,1))
conv1=tf.layers.conv2d(x,filters=64,kernel_size=[3,3],strides=[1,1],kernel_regularizer=tf.contrib.layers.l1_regularizer(0.01))