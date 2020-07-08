'''
@Title: tf_data基础api的使用
@Date:20200512
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

#from_tensor_slices该函数是dataset核心函数之一，它的作用是把给定的元组、列表和张量等数据进行特征切片。
#切片的范围是从最外层维度开始的。如果有多个特征进行组合，那么一次切片是把每个组合的最外维度的数据切开，分成一组一组的。
# https://blog.csdn.net/Dr_jiabin/article/details/93366661

dataset =  tf.data.Dataset.from_tensor_slices(np.arange(10))
print(dataset)
# <TensorSliceDataset shapes: (), types: tf.int32>
for item in dataset:
    print(item)
    '''
    tf.Tensor(0, shape=(), dtype=int32)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)
    tf.Tensor(4, shape=(), dtype=int32)
    tf.Tensor(5, shape=(), dtype=int32)
    tf.Tensor(6, shape=(), dtype=int32)
    tf.Tensor(7, shape=(), dtype=int32)
    tf.Tensor(8, shape=(), dtype=int32)
    tf.Tensor(9, shape=(), dtype=int32)
    '''

# 1.repeat epoch
# 2.get batch
dataset = dataset.repeat(3).batch(7)
print(dataset)
for item in dataset:
    print(item)
print("**************dataset")
# interleave:
# case:文件dataset -> 具体数据集
dataset2 =  dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),# map_fn
    cycle_length = 5, # cycle_length
    block_length = 5 # block_length
)
for item in dataset2:
    print(item)

x = np.array([[1,2],[3,4],[5,6]])
y = np.array(["cat","dog","fox"])
dataset3 = tf.data.Dataset.from_tensor_slices((x,y))#输入元组
print(dataset3)

for item_x,item_y in dataset3:
    print(item_x,item_y)
    print(item_x.numpy(),item_y.numpy())

dataset4 = tf.data.Dataset.from_tensor_slices({"feature":x,
                                               "label":y})#输入字典
for item in dataset4:
    print(item["feature"].numpy(),item["label"].numpy())

print("**************************************************8")
np.random.seed(0)
x1 = np.random.sample((11,2))
# make a dataset from a numpy array
print(x1)

dataset5 = tf.data.Dataset.from_tensor_slices(x1)
for item in dataset5:
    print(item)
print("******dataset5")
dataset6 = dataset5.shuffle(2)  # 将数据打乱，数值越大，混乱程度越大
for item in dataset6:
    print(item)
print("******dataset6")
dataset7 = dataset5.batch(4)  # 按照顺序取出4行数据，最后一次输出可能小于4
print(dataset7)
for item in dataset7:
    print(item)
print("******dataset7")
dataset8 = dataset5.repeat(2)  # 数据集重复了指定次数
for item in dataset8:
    print(item)
print("******dataset8")
# repeat()在batch操作输出完毕后再执行,若在之前，相当于先把整个数据集复制两次
#为了配合输出次数，一般默认repeat()空


