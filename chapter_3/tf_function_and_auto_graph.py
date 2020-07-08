'''
Title: tf.function函数转换:1.将python函数转换为tensorflow的图结构，优化并加速我们自己编写的python函数 2.函数签名与图结构
Date:20200508
Author:Bello
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import sys
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime



print(tf.__version__)
print(sys.version_info)
for module in mpl,np,pd,sklearn,tf,keras,cv:
    print(module.__name__,module.__version__)

#tf.function and auto_graph
def scaled_elu(z,scale = 1.0,alpha = 1.0):
    #z > 0 ? scale * z : scale * alpha * tf.nn.elu(z)
    is_positive = tf.greater(z,0.0)
    return scale * tf.where(is_positive,z,alpha * tf.nn.elu(z))
print(scaled_elu(tf.constant(-3.))) #标量
print(scaled_elu(tf.constant([-3.,-2.5])))#向量
#第一种方式：tf.function()
scaled_elu_tf = tf.function(scaled_elu)
print(scaled_elu_tf(tf.constant([-3.])))
print(scaled_elu_tf(tf.constant([-3,-2.5])))
print(scaled_elu_tf.python_function is scaled_elu)#验证scaled_wlu_tf 的 源pyhton函数是否是scaled_elu


start = datetime.datetime.now()
scaled_elu(tf.random.normal((1000,1000)))
end = datetime.datetime.now()
print (end-start)

start1 = datetime.datetime.now()
scaled_elu_tf(tf.random.normal((1000,1000)))
end1 = datetime.datetime.now()
print (end1-start1)

#第二种方式：@tf.function
#1 + 1/2 + 1/2^2 + ... + 1/2^n
#@tf.function
def converge_to_2(n_iters):
    total = tf.constant(0.)
    increment = tf.constant(1.)
    for _ in range(n_iters):
        total += increment
        increment /= 2.0
        return total
print(converge_to_2(20))


# 第三种：这个函数可以把普通的python函数转成tf代码，这个tf代码就可以生成图结构
def display_tf_code(func):
    code = tf.autograph.to_code(func) #tf1.0
    print(code)
    from IPython.display import display,Markdown
    display(Markdown('```pyhton\n{}\n```'.format(code)))
display_tf_code(scaled_elu)
display_tf_code(converge_to_2)


var = tf.Variable(0.)#如果有Variable，则必须定义在tf.function的外面
@tf.function
def add_21():
    return var.assign_add(21)#+=
print(add_21())


@tf.function
def cube(z):
    return tf.pow(z,3)
print(cube(tf.constant([1.,2.,3.])))
print(cube(tf.constant([1,2,3])))
print("**********************************")

@tf.function(input_signature=[tf.TensorSpec([None],tf.int32,name = 'x')])
def cube(z):
    return tf.pow(z,3)
try:
    print(cube(tf.constant([1.,2.,3.])))
except ValueError as ex:
    print(ex)

print(cube(tf.constant([1,2,3])))
print("**************************************")
#  https://blog.csdn.net/l7h9ja4/article/details/92857163 详细内容参考

#@tf.function py func -> tf graph
#get_concrete_function -> add @tf.function py func with input signature -> SavedModel
cube_func_int32 = cube.get_concrete_function(tf.TensorSpec([None],tf.int32))
print(cube_func_int32)

print(cube_func_int32 is cube.get_concrete_function(tf.TensorSpec([5],tf.int32)))
print(cube_func_int32 is cube.get_concrete_function(tf.constant([1,2,3])))
print(cube.get_concrete_function(tf.constant([1,2,3]))
      is cube.get_concrete_function(tf.TensorSpec([5],tf.int32)))#True 说明signature一样！

print(cube_func_int32.graph)
print(cube_func_int32.graph.get_operations())

pow_op = cube_func_int32.graph.get_operations()[2]
print(pow_op)
print(list(pow_op.inputs))
print(list(pow_op.outputs))
print(cube_func_int32.graph.get_operation_by_name("x"))
print(cube_func_int32.graph.get_tensor_by_name("x:0"))
print(cube_func_int32.graph.as_graph_def())
print("****************************************************")
print(help(tf.Graph().get_operations))


