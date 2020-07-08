'''
Title: 手动实现超参数搜索
Date:20200506
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

#learning_rate:[1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]
#w = w + grad * learning_rate
leraning_rates = [1e-4,3e-4,1e-3,3e-3,1e-2,3e-2]
histories = []
for lr in leraning_rates:
    model = keras.models.Sequential([
        keras.layers.Dense(30,activation='relu',input_shape=x_train_scaled.shape[1:]),
        keras.layers.Dense(1)
    ])
    optimizer = keras.optimizers.SGD(lr)
    model.compile(loss='mean_squared_error',
                  optimizer=optimizer)
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)#关注验证机上目标函数的值
    ]
    history = model.fit(x_train_scaled,y_train,
                        epochs=100,
                        validation_data=(x_valid_scaled,y_valid),
                        callbacks=callbacks
                        )
    histories.append(history)
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
for lr,history in zip(leraning_rates,histories):
    print("Learning_rate: ",lr)
    plot_learning_curves(history)









