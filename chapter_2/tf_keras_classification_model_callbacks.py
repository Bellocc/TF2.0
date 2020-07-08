'''
@Title:tf_keras_classification_model_callbacks
@Date:20200427
@Author:Bello
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import os
import sklearn
import time
import tensorflow as  tf
from tensorflow import  keras
import gzip
from sklearn.preprocessing import StandardScaler

# print(tf.__verdion__)
# print(sys.version_info)
# for module in mpl,np,pd,sklearn,tf,keras:
#     print(module.__name__,module.__version__)

def load_data(data_folder):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]
    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder,fname))

    with gzip.open(paths[0],'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(),np.uint8,offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train),28,28)

    with gzip.open(paths[2],'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(),np.uint8,offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test),28,28)

    return (x_train,y_train),(x_test,y_test)

(x_train_all, y_train_all), (x_test, y_test) = load_data('./Fashion_Mnist')
x_valid,x_train = x_train_all[:5000],x_train_all[5000:]
y_valid,y_train = y_train_all[:5000],y_train_all[5000:]
print(x_train.shape,y_train.shape)
print(x_valid.shape,y_valid.shape)
print(x_test.shape,y_test.shape)

#归一化：x = (x - u) / std
scalar = StandardScaler()
#X_train:[None,28,28] - > [None,784]
x_train_scaled = scalar.fit_transform(
    x_train.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_valid_scaled = scalar.transform(
    x_valid.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
x_test_scaled = scalar.transform(
    x_test.astype(np.float32).reshape(-1,1)).reshape(-1,28,28)
#tf.keras.models.Sequential()
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28]))
model.add(keras.layers.Dense(300,activation = 'relu'))
#参数量：[None,784] * w + b  ->[None,300]  w.shape = [784,300],b.shape = [300] 参数量为784*300+300
model.add(keras.layers.Dense(100,activation = 'relu'))
model.add(keras.layers.Dense(10,activation='softmax'))
'''
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape = [28,28]),
    keras.layers.Dense(300,activation = 'relu'),
    keras.layers.Dense(100,activation = 'relu'),
    keras.layers.Dense(10,activation='softmax')
])
'''



#relu: y = max(0,x)
#softmax: 将向量变成概率分布.x = [x1,x2,x3]
#         y = [e^x1/sum,e^x2/sum,e^x3/sum],sum = e^x1+e^x2+e^x3
#reason for sparse:y->index(sparse_categorical_crossentropy).y->onehot(categorical_crossentropy)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer = 'sgd',
              metrics=['accuracy'])

#Tensorboard , Earlystopping, ModelCheckpoint
logdir = 'callbacks'
if not os.path.exists(logdir):
    os.mkdir(logdir)
output_model_file = os.path.join(logdir,'fashion_mnist_model.h5')
callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only=True),#默认最近的，best表示保存最好的。
    keras.callbacks.EarlyStopping(patience=5,min_delta=1e-3)#关注验证集上目标函数的值
]
history = model.fit(x_train_scaled,y_train,epochs=100,
                    validation_data=(x_valid_scaled,y_valid),
                    callbacks=callbacks)

def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)#显示网格
    plt.gca().set_ylim(0,1)
    plt.show()
plot_learning_curves(history)
print(history.history)

model.evaluate(x_test_scaled,y_test)
