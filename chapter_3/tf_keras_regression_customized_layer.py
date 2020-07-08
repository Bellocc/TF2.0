'''
Title: 自定义Denselayer
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

'''
layer = tf.keras.layers.Dense(100)
layer = tf.keras.layers.Dense(100,input_shape = (None,5))
layer(tf.zeros([10,5]))

layer.Variables():可以打印出这个layer中包含的所有参数
x * w + b 
layer.trainable_variables:layer中所有可训练的变量
help(layer)

'''
#tf.nn.softplus:log(1+e^x)
customized_softplus = keras.layers.Lambda(lambda x:tf.nn.softplus(x))

#customized dense layer.
class CustomizedDenseLayer(keras.layers.Layer):
    def __init__(self,units,activation = None,**kwargs):
        self.units = units
        self.activation = keras.layers.Activation(activation)
        super(CustomizedDenseLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        # x * w + b  input_shape:[None,a] w:[a,b] output_shape:[None,b]
        self.kernel = self.add_weight(name = 'kernel',
                                      shape = (input_shape[1],self.units),
                                      initializer = 'uniform',
                                      trainable = True)
        self.bias = self.add_weight(name = 'bias',
                                    shape = (self.units,),
                                    initializer = 'zeros',
                                    trainable = True)
        super(CustomizedDenseLayer,self).build(input_shape) #调用父类的build函数
    def call(self, x):
        """完成正向计算"""
        return self.activation(x @ self.kernel + self.bias)

model = keras.models.Sequential([
    CustomizedDenseLayer(30,activation = 'relu',input_shape = x_train_scaled.shape[1:]),
    CustomizedDenseLayer(1),
    customized_softplus,
    #等价于:keras.layers.Dense(1,activation = 'softplus'),
    #等价于:keras.layers,Dense(1),keras.layers.Activation('softplus')
])
model.summary()
model.compile(loss='mean_squared_error',
              optimizer='sgd')
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









