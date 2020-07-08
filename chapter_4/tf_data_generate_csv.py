'''
@Title: 生成csv文件，tf.io.decode_csv使用,tf.data读取csv文件并与tf.keras结合使用
@Date:20200513
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

output_dir = "generate_csv"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
def save_to_dir(output_dir,data,name_prefix,header=None,n_parts=10):
    path_format = os.path.join(output_dir,"{}_{:02d}.csv")
    filenames = []
    for file_idx,row_indices in enumerate(np.array_split(np.arange(0,len(data)),n_parts)):
        part_csv = path_format.format(name_prefix,file_idx)
        filenames.append(part_csv)
        with open(part_csv,"wt",encoding="utf-8") as f:
            if header is not None:
                f.write(header + "\n")
            for row_index in row_indices:
                f.write(",".join(repr(col) for col in data[row_index]))#repr()返回一个对象的 string 格式。
                f.write("\n")
    return filenames

train_data = np.c_[x_train_scaled,y_train]
valid_data = np.c_[x_valid_scaled,y_valid]
test_data = np.c_[x_test_scaled,y_test]
header_cols = housing.feature_names + ["MidianHouseValue"]
header_str = ",".join(header_cols)# join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。

train_filenames = save_to_dir(output_dir,train_data,"train",header_str,n_parts=20)
valid_filenames = save_to_dir(output_dir,valid_data,"valid",header_str,n_parts=10)
test_filenames = save_to_dir(output_dir,test_data,"test",header_str,n_parts=10)

import pprint
print("train filenames:")
pprint.pprint(train_filenames)
print("valid filenames:")
pprint.pprint(valid_filenames)
print("test filenames:")
pprint.pprint(test_filenames)

# 1. filenames -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv
fielname_dataset = tf.data.Dataset.list_files(train_filenames)#返回值：文件路径或者文件路径列表的Dataset形式
for filename in fielname_dataset:
    print(filename)

n_readers = 5
dataset = fielname_dataset.interleave(
    #fielname_dataset.interleave()
    #参数：1.dataset:输入的dataset形式的值；2.function:经过的函数function
    # 返回值：通过function函数产生的返回值。
    lambda filename:tf.data.TextLineDataset(filename).skip(1),#文件路径或者文件路径列表 按行排列的 字符串Dataset形式,Dataset中的每一个元素就对应了文件中的一行
    cycle_length=n_readers
)
for line in dataset.take(15):
    print(line.numpy())

# tf.io.decode_csv(str,record_default)
sample_str = '1,2,3,4,5'
record_default = [tf.constant(0,dtype=tf.int32)] * 5
parsed_fields = tf.io.decode_csv(sample_str,record_default)
pprint.pprint(parsed_fields)
'''
[<tf.Tensor: shape=(), dtype=int32, numpy=1>,
 <tf.Tensor: shape=(), dtype=int32, numpy=2>,
 <tf.Tensor: shape=(), dtype=int32, numpy=3>,
 <tf.Tensor: shape=(), dtype=int32, numpy=4>,
 <tf.Tensor: shape=(), dtype=int32, numpy=5>]
'''
sample_str1 = '1,2,3,4,5'
record_default = [
    tf.constant(0,dtype=tf.int32),
    0,
    np.nan,
    "hello",
    tf.constant([])
]
parsed_fields1 = tf.io.decode_csv(sample_str1,record_default)#Tensor对象列表。与record_defaults具有相同的类型。每个张量将与记录具有相同的形状。
pprint.pprint(parsed_fields1)
'''
[<tf.Tensor: shape=(), dtype=int32, numpy=1>,
 <tf.Tensor: shape=(), dtype=int32, numpy=2>,
 <tf.Tensor: shape=(), dtype=float32, numpy=3.0>,
 <tf.Tensor: shape=(), dtype=string, numpy=b'4'>,
 <tf.Tensor: shape=(), dtype=float32, numpy=5.0>]
'''

try:
    parsed_fields2 = tf.io.decode_csv(",,,,",record_default)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

try:
    parsed_fields2 = tf.io.decode_csv("1,2,3,4,5,6,7,8,9",record_default)
except tf.errors.InvalidArgumentError as ex:
    print(ex)

def parse_csv_line(line,n_fields = 9):
    defs = [tf.constant(np.nan)] * n_fields
    parsed_fields = tf.io.decode_csv(line,record_defaults = defs)
    x = tf.stack(parsed_fields[0:-1])
    y = tf.stack(parsed_fields[-1:])
    return x , y
print(parse_csv_line(b'-1.4008505650808356,-0.2136274401485836,-0.6933017675424598,0.004759191508236944,-0.5087275363785142,0.070549953427866,-0.7508635889055536,0.6982597472157485,1.438',
               n_fields = 9))

# tf.data读取csv文件综合
# 1. filenames -> dataset
# 2. read file -> dataset -> datasets -> merge
# 3. parse csv
def csv_reader_dataset(filenames, n_readers=5,
                       batch_size=32, n_parse_threads=5,
                       shuffle_buffer_size=10000):
    dataset = tf.data.Dataset.list_files(filenames)
    dataset = dataset.repeat()
    dataset = dataset.interleave(
        lambda filename: tf.data.TextLineDataset(filename).skip(1),
        cycle_length = n_readers
    )
    dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(parse_csv_line,
                          num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset

train_set = csv_reader_dataset(train_filenames, batch_size=3)
for x_batch, y_batch in train_set.take(2):
    print("x:")
    pprint.pprint(x_batch)
    print("y:")
    pprint.pprint(y_batch)

batch_size = 32
train_set = csv_reader_dataset(train_filenames,batch_size=batch_size)
valid_set = csv_reader_dataset(valid_filenames,batch_size=batch_size)
test_set = csv_reader_dataset(test_filenames,batch_size=batch_size)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation='relu',
                       input_shape=[8]),
    keras.layers.Dense(1),
])
model.compile(loss="mean_squared_error", optimizer="sgd")
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]

history = model.fit(train_set,
                    validation_data = valid_set,
                    steps_per_epoch = 11160 // batch_size,
                    validation_steps = 3870 // batch_size,
                    epochs = 100,
                    callbacks = callbacks)
model.evaluate(test_set,steps = 5160 // batch_size)











