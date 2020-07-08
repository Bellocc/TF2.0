'''
@Title:Feature_column:
在TensorFlow中，特征列(Feature column)是原始数据和 Estimator 之间的接口，它告诉Estimator如何使用数据。
神经网络接受的输入，只能是数值，而且是整理好的数值
所以，原始数据 和 神经网络输入需求之间需要一个桥梁，这个桥梁就是特征列(Feature column)
@Author:Bello
@Date:2020/05/28
https://blog.csdn.net/wwangfabei1989/article/details/90754536
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
train_dataset = make_dataset(train_df,y_train,batch_size=5)
print(train_dataset)
for x,y in train_dataset.take(1):
    print(x,y)

# keras.layers.DenseFeatures
# feature_column 本质上是一组对Feature变换的规则
# https://blog.csdn.net/wwangfabei1989/article/details/90754536
for x,y in train_dataset.take(1):
    age_column = feature_columns[7]
    gender_column = feature_columns[0]
    print(keras.layers.DenseFeatures(age_column)(x).numpy())
    print(keras.layers.DenseFeatures(gender_column)(x).numpy())
    print(keras.layers.DenseFeatures(feature_columns)(x).numpy())

model = keras.models.Sequential([
    keras.layers.DenseFeatures(feature_columns),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(2,activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy']
)
# 1.model.fit()
# 2.model -> estimator -> train
'''
train_dataset = make_dataset(train_df,y_train,epochs = 100)
eval_dataset = make_dataset(eval_df,y_eval,epochs=1,shuffle=False)
history = model.fit(train_dataset,
                    validation_data=eval_dataset,
                    steps_per_epoch=20,
                    validation_steps=8,
                    epochs=100)
'''
# train_df有627条数据，repeat100之后62700条数据，设置batch_size=32，一共有1960个batch
# 当设置steps_per_epoch = 20 时，一共1960个batch，每个epoch便不再重新遍历训练集，即每个epoch只取20个batch训练。
# 当执行到98个epoch（98*20=1960），报错已经取不出数据了
# 如果想要将epochs跑完，在数据repeat的时候，括号不填将其无限复制
estimator = keras.estimator.model_to_estimator(model)
# 1.function
# 2.return a.(features,labels) b.dataset -> (feature,label)
estimator.train(input_fn = lambda : make_dataset(train_df,y_train,epochs=100))# tf2.0 Bug



