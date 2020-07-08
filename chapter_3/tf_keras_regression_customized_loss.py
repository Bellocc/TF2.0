'''
Title: 自定义损失函数与DenseLayer
Date:20200508
Author:Bello
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

def customized_mse(y_true,y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

model = keras.models.Sequential([
    keras.layers.Dense(30,activation='relu',input_shape=x_train_scaled.shape[1:]),
    keras.layers.Dense(1)
])
model.summary()
model.compile(loss=customized_mse,
              optimizer='sgd',
              metrics=['mean_squared_error'])
callbacks = [
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)#关注验证机上目标函数的值
]
history = model.fit(x_train_scaled,y_train,epochs=100,
                    validation_data=(x_valid_scaled,y_valid),
                    callbacks=callbacks
                    )

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)
model.evaluate(x_test_scaled,y_test)









