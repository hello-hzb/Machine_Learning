# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from x1_perceptron import Perceptron, PerceptronDual

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.normal(0, 1, size=(100, 2))) # 随机输入
y_data = np.dot(x_data, [0.500, 0.500]) - 0.700
y_data[y_data >= 0] = 1
y_data[y_data < 0] = -1

percept = Perceptron(2, 0.01, 100)
# print percept.predict(x_data)
#
percept.fit(x_data, y_data)
#
print(percept.weight)
print(percept.bias)
