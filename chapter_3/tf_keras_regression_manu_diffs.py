'''
@Title: tf.GradientTape与tf.keras结合使用
@Date:20200508
@Author:Bello
'''

import  matplotlib as mpl
import  matplotlib.pyplot as plt
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



'''
print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras:
    print(module.__name__,module.__version__)
'''
housing =  fetch_california_housing()
print(housing.DESCR)
print(housing.data.shape)
print(housing.target.shape)

x_train_all,x_test,y_train_all,y_test = train_test_split(housing.data,housing.target,random_state=3)
x_train,x_valid,y_train,y_valid = train_test_split(x_train_all,y_train_all,random_state=11)
print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# metric 使用
metric = keras.metrics.MeanSquaredError()
print(metric([5.],[2.]))
print(metric([0.],[1.]))
print(metric.result())# 默认会累加数据
# 如果不想累加误差
metric.reset_states()
metric([1.],[3.])
print(metric.result())
'''
tf.Tensor(9.0, shape=(), dtype=float32)
tf.Tensor(5.0, shape=(), dtype=float32)
tf.Tensor(5.0, shape=(), dtype=float32)
tf.Tensor(4.0, shape=(), dtype=float32)
'''

#1. batch 遍历训练集 metric
#   1.1 自动求导
#2. epoch结束 验证集 metric
epochs = 100
batch_size = 32
steps_per_epch = len(x_train_scaled) // batch_size
optimizer = keras.optimizers.SGD()
metric = keras.metrics.MeanSquaredError()

def random_batch(x,y,batch_size = 32):
    idx = np.random.randint(0,len(x),size = batch_size)
    return x[idx],y[idx]

model = keras.models.Sequential([
    keras.layers.Dense(30,activation='relu',input_shape=x_train_scaled.shape[1:]),
    keras.layers.Dense(1)
])

for epoch in range(epochs):
    metric.reset_states()
    for step in range(steps_per_epch):
        x_batch,y_batch = random_batch(x_train_scaled,y_train,batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_pred,y_batch))
            metric(y_pred,y_batch)
        grads = tape.gradient(loss,model.variables)
        grads_and_vars = zip(grads,model.variables)
        optimizer.apply_gradients(grads_and_vars)
        print("\rEpoch",epoch,"train mse:",metric.result().numpy(),end="")
    y_valid_pred = model(x_valid_scaled)
    valid_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_valid,y_valid_pred))
    print("\t","valid_loss:",valid_loss.numpy())










