'''
Title:近似求导 + tf.GradientTape基本使用方法
Date:20200509
Author:Bello
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime

# print(tf.__version__)
# print(sys.version_info)
# for module in mpl,np,pd,sklearn,tf,keras:
#     print(module.__name__,module.__version__)

def f(x):
    return 3. * x ** 2 + 2.* x - 1
def approximate_derivative(f,x,eps=1e-3):
    return (f(x + eps)-f(x - eps)) / (2.*eps)
print(approximate_derivative(f,1.))

def g(x1,x2):
    return (x1 + 5) * (x2 ** 2)
def approximate_gradient(g,x1,x2,eps=1e-3):
    dg_x1 = approximate_derivative(lambda x:g(x,x2),x1,eps)
    dg_x2 = approximate_derivative(lambda x:g(x1,x),x2,eps)
    return dg_x1,dg_x2

print(approximate_gradient(g,2.,3.))

# tf.GradientTape
# x1 = tf.Variable(2.)
# x2 = tf.Variable(3.)
# with tf.GradientTape() as tape:
#     z = g(x1,x2)
# dz_x1 = tape.gradient(z,x1)#tape只能用一次，就会被消减(GradientTape.gradient can only be called once on non-persistent tapes.)
# print(dz_x1)
#
# try:
#     dz_x2 = tape.gradient(z,x2)
# except RuntimeError as ex:#GradientTape.gradient can only be called once on non-persistent tapes.
#     print(ex)
#
#为了解决tape只能用一次的问题
# x1 = tf.Variable(2.)
# x2 = tf.Variable(3.)
# with tf.GradientTape(persistent = True) as tape:#此时tape不会自己消除，必须手动消除
#     z = g(x1,x2)
# dz_x1 = tape.gradient(z,x1)
# dz_x2 = tape.gradient(z,x2)
# print(dz_x1,dz_x2)
# del tape


#如何实现一次求出x1,x2的偏导数呢
# x1 = tf.Variable(2.)
# x2 = tf.Variable(3.)
# with tf.GradientTape() as tape:
#     z = g(x1,x2)
# dz_x1x2 = tape.gradient(z,[x1,x2])
# print(dz_x1x2)
# [<tf.Tensor: shape=(), dtype=float32, numpy=9.0>, <tf.Tensor: shape=(), dtype=float32, numpy=42.0>]


#如何关注constant的梯度
# x1 = tf.constant(2.)
# x2 = tf.constant(3.)
# with tf.GradientTape() as tape:
#     z = g(x1,x2)
# dz_x1x2 = tape.gradient(z,[x1,x2])
# print(dz_x1x2)
# [None,None]

# x1 = tf.constant(2.)
# x2 = tf.constant(3.)
# with tf.GradientTape() as tape:
#     tape.watch(x1)
#     tape.watch(x2)
#     z = g(x1,x2)
# dz_x1x2 = tape.gradient(z,[x1,x2])
# print(dz_x1x2)
# print(x1,x2)
# [<tf.Tensor: shape=(), dtype=float32, numpy=9.0>, <tf.Tensor: shape=(), dtype=float32, numpy=42.0>]
# tf.Tensor(2.0, shape=(), dtype=float32) tf.Tensor(3.0, shape=(), dtype=float32)

#如何实现两个函数对一个变量求偏导
# x = tf.Variable(5.0)
# with tf.GradientTape() as tape:
#     z1 = 3 * x
#     z2 = x ** 2
# dz1z2_x = tape.gradient([z1,z2],x)#求出来的是z1对x的导数与z2对x的导数的和
# print(dz1z2_x)
# tf.Tensor(13.0, shape=(), dtype=float32)


#如何求二阶偏导数
# x1 = tf.Variable(2.0)
# x2 = tf.Variable(3.0)
# with tf.GradientTape(persistent=True) as outer_tape:
#     with tf.GradientTape(persistent=True) as inner_tape:
#         z = g(x1,x2)
#     inner_grads = inner_tape.gradient(z,[x1,x2])
# outer_grads = [outer_tape.gradient(inner_grad,[x1,x2]) for inner_grad in inner_grads]
# print(outer_grads)
# del inner_tape
# del outer_tape
# [[None, <tf.Tensor: shape=(), dtype=float32, numpy=6.0>],
# [<tf.Tensor: shape=(), dtype=float32, numpy=6.0>, <tf.Tensor: shape=(), dtype=float32, numpy=14.0>]]
# [[a^2z/ax1^2,a^2z/ax2ax1],[a^2z/ax1ax2,a^2z/ax2^2]

#梯度下降的模拟
# learning_rate = 0.1
# x = tf.Variable(0.0)
# for  _ in range(100):
#     with tf.GradientTape() as tape:
#         z = f(x)
#     dz_dx = tape.gradient(z,x)
#     x.assign_sub(learning_rate * dz_dx)
# print(x)
# <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-0.3333333>

learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr=learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
        dz_dx = tape.gradient(z,x)
        optimizer.apply_gradients([(dz_dx,x)])
print(x)
