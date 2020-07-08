# import tensorflow as tf
# print(tf.__version__)
# x = tf.constant(0.)
# y = tf.constant(1.0)
# for interation in range(50):
#     x = x + y
#     y = y / 2
# print(x.numpy())

# a = ["1","2","3"]
# b = ["4"]
# c = a + b
# d = ",".join(c)
# print(c)
# print(d)

import  matplotlib as mpl
import  matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import keras
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras:
    print(module.__name__,module.__version__)
'''
2.1.0
sys.version_info(major=3, minor=6, micro=5, releaselevel='final', serial=0)
matplotlib 2.2.2
numpy 1.18.2
pandas 0.23.0
sklearn 0.19.1
tensorflow 2.1.0
tensorflow_core.python.keras.api._v2.keras 2.2.4-tf
'''
