'''
Title: wide_deep模型的多输出
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

#多输入
input_wide = keras.layers.Input(shape=[5])
input_deep = keras.layers.Input(shape=[6])
hidden1 = keras.layers.Dense(30,activation='relu')(input_deep)
hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)
concat = keras.layers.concatenate([input_wide,hidden2])
output = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(hidden2)
model = keras.models.Model(inputs = [input_wide,input_deep],
                           outputs = [output,output2])

model.summary()
model.compile(loss='mean_squared_error',
              optimizer='sgd')
callbacks = [
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)#关注验证机上目标函数的值
]

x_train_scaled_wide = x_train_scaled[:,:5]
x_train_scaled_deep = x_train_scaled[:,2:]
x_valid_scaled_wide = x_valid_scaled[:,:5]
x_valid_scaled_deep = x_valid_scaled[:,2:]
x_test_scaled_wide = x_test_scaled[:,:5]
x_test_scaled_deep = x_test_scaled[:,2:]

history = model.fit([x_train_scaled_wide,x_train_scaled_deep],
                    [y_train,y_train],
                    epochs=100,
                    validation_data=([x_valid_scaled_wide,x_valid_scaled_deep],
                                     [y_valid,y_valid]),
                    callbacks=callbacks
                    )

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)

model.evaluate([x_test_scaled_wide,x_test_scaled_deep],
               [y_test,y_test])








