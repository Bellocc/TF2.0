'''
Title: sklearn封装keras模型,定义参数集合,搜索参数
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
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV


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

#RandomizedSearchCV
# 1. 转化为sklearn的model
# 2. 定义参数集合
# 3. 搜索参数
def build_model(hidden_layers = 1,
                layer_size = 30,
                learning_rate = 3e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(layer_size,activation='relu',input_shape=x_train.shape[1:]))
    for _ in range(hidden_layers - 1):
        model.add(keras.layers.Dense(layer_size,activation='relu'))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss='mse',optimizer=optimizer)
    return model
sklearn_model = keras.wrappers.scikit_learn.KerasRegressor(build_model)
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)]#关注验证机上目标函数的值

#f(x) = 1 / (x * log(b / a)) a <= x <= b
param_distribution = {
    "hidden_layers":[1,2,3,4],
    "layer_size":np.arange(1,100),
    "learning_rate":reciprocal(1e-4,1e-2)
}

random_search_cv = RandomizedSearchCV(sklearn_model,
                                      param_distribution,
                                      n_iter=10,#多少组参数集合
                                      cv = 3,
                                      n_jobs=1)#有多少任务在并行处理
random_search_cv.fit(x_train_scaled,y_train,epochs=100,
                     validation_data=(x_valid_scaled,y_valid),
                     callbacks=callbacks)
# cross_validation:训练集分成n份，n-1训练，最后一份验证
print(random_search_cv.best_params_)
print(random_search_cv.best_score_)
print(random_search_cv.best_estimator_)

model = random_search_cv.best_estimator_.model
model.evaluate(x_test_scaled,y_test)












