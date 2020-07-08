'''
Title: 函数式API(功能API)实现wide_deep模型
Date:20200505
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
print(x_train_scaled.shape[1:])

#函数式API，功能API
input = keras.layers.Input(shape=x_train_scaled.shape[1:])
hidden1 = keras.layers.Dense(30,activation='relu')(input)
hidden2 = keras.layers.Dense(30,activation='relu')(hidden1)
#复合函数：f(x) = h(g(x))
concat = keras.layers.concatenate([input,hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs = [input],
                           outputs = [output])



model.summary()
model.compile(loss='mean_squared_error',
              optimizer='sgd')

logdir = 'wide_deep_regression_callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,'fetch_california_housing_model.h5')
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only=True), #默认最近的，best表示模型只保存最好的
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








