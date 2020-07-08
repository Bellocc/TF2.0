'''
@Title:预定义estimator使用
@Author:Bello
@Date:2020/06/03
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras
'''
print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)
'''
train_file = "./data/titanic/train.csv"
eval_file = "./data/titanic/eval.csv"

train_df = pd.read_csv(train_file)
eval_df = pd.read_csv(eval_file)
y_train = train_df.pop('survived')
y_eval = eval_df.pop('survived')
print(train_df.head())
print(eval_df.head())

categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class',
                       'deck', 'embark_town', 'alone']
numeric_columns = ['age', 'fare']
feature_columns = []
#离散型：
for categorical_column in categorical_columns:
    vocab = train_df[categorical_column].unique()
    print(categorical_column,vocab)
    feature_columns.append(
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                categorical_column,vocab)))

#连续型：
for categorical_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(
            categorical_column,dtype=tf.float32))
print("********************")
print(feature_columns)

def make_dataset(data_df,label_df,epochs = 10,batch_size = 32,shuffle = True):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
    if shuffle:
        dataset = dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batch_size)
    #按照顺序依次取出32行数据(tf.Tensor)，最后一次输出可能小于32，最后组合成一个BatchDataset，Batch=n/batch_size
    return dataset

# 基线分类器
# output_dir = 'baseline_model'
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# baseline_estimator = tf.estimator.BaselineClassifier(model_dir = output_dir,
#                                                      n_classes = 2)
# baseline_estimator.train(input_fn = lambda :make_dataset(train_df,y_train,epochs = 100))
# baseline_estimator.evaluate(input_fn = lambda :make_dataset(eval_df,y_eval,epochs = 1,shuffle = False,batch_size = 20))

# 线性分类器
# linear_output_dir = 'linear_model'
# if not os.path.exists(linear_output_dir):
#     os.mkdir(linear_output_dir)
# linear_estimator = tf.estimator.LinearClassifier(model_dir = linear_output_dir,
#                                                   n_classes = 2,
#                                                   feature_columns = feature_columns)#feature_columns 解析dataset
# linear_estimator.train(input_fn = lambda :make_dataset(train_df,y_train,epochs = 100))
# linear_estimator.evaluate(input_fn = lambda :make_dataset(eval_df,y_eval,epochs = 1,shuffle = False))
# Process finished with exit code -1073740791 (0xC0000409)
# 显存不足，导致程序没有成功，在logdir DIRECTORY_PATH中没有找到事件文件，
# 即生成的日志文件有问题，才导致没有scalar显示，不是tensorboard的问题

# DNNClassifier
dnn_output_dir = './dnn_model'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)
dnn_estimator = tf.estimator.DNNClassifier(model_dir = dnn_output_dir,
                                           n_classes = 2,
                                           feature_columns = feature_columns,
                                           hidden_units = [128,128],
                                           activation_fn = tf.nn.relu,
                                           optimizer = 'Adam')
dnn_estimator.train(input_fn = lambda :make_dataset(train_df,y_train,epochs = 100))
dnn_estimator.evaluate(input_fn = lambda :make_dataset(eval_df,y_eval,epochs = 1,shuffle = False))



