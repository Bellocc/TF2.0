'''
@Title:交叉特征实战
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
for numeric_column in numeric_columns:
    feature_columns.append(
        tf.feature_column.numeric_column(
            numeric_column,dtype=tf.float32))
print("********************")
print(feature_columns)

# cross feature: age: [1,2,3,4,5], gender:[male, female]
# age_x_gender: [(1, male), (2, male), ..., (5, male), ..., (5, female)]
# 100000: 100 -> hash(100000 values) % 100
feature_columns.append(
    tf.feature_column.indicator_column(
        tf.feature_column.crossed_column(
            ['age','sex'],hash_bucket_size = 100)))

def make_dataset(data_df,label_df,epochs = 10,shuffle = True,batchsize = 32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(data_df),label_df))
    if shuffle:
        dataset.shuffle(10000)
    dataset = dataset.repeat(epochs).batch(batchsize)
    return dataset

# 基线分类器
# output_dir = 'baseline_model_new_features'
# if not os.path.exists(output_dir):
#     os.mkdir(output_dir)
# baseline_estimator = tf.estimator.BaselineClassifier(model_dir = output_dir,
#                                                      n_classes = 2)
# baseline_estimator.train(input_fn = lambda :make_dataset(train_df,y_train,epochs = 100))
# baseline_estimator.evaluate(input_fn = lambda :make_dataset(eval_df,y_eval,epochs = 1,shuffle = False,batchsize = 20))

# 线性分类器
# linear_output_dir = 'linear_model_new_features'
# if not os.path.exists(linear_output_dir):
#     os.mkdir(linear_output_dir)
# linear_estimator = tf.estimator.LinearClassifier(model_dir = linear_output_dir,
#                                                  n_classes = 2,
#                                                  feature_columns = feature_columns)
# linear_estimator.train(input_fn = lambda : make_dataset(train_df, y_train, epochs = 100))
# linear_estimator.evaluate(input_fn = lambda : make_dataset(eval_df, y_eval, epochs = 1, shuffle = False))

# DNNclassifier
dnn_output_dir = './dnn_model_new_features'
if not os.path.exists(dnn_output_dir):
    os.mkdir(dnn_output_dir)
dnn_estimator = tf.estimator.DNNClassifier(model_dir = dnn_output_dir,
                                           n_classes = 2,
                                           feature_columns=feature_columns,
                                           hidden_units = [128, 128],
                                           activation_fn = tf.nn.relu,
                                           optimizer = 'Adam')
dnn_estimator.train(input_fn = lambda : make_dataset(train_df, y_train, epochs = 100))
dnn_estimator.evaluate(input_fn = lambda : make_dataset(eval_df, y_eval, epochs = 1, shuffle = False))

