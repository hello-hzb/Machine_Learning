#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division   # 为了能够用/得到标准的除法结果
import numpy as np


class Perceptron(object):
    """感知机的标准实现"""
    def __init__(self,
                 feature_size,
                 lr_init,
                 epoch,
                 activation=None,
                 lr_decay_rate=None,
                 mu=0,
                 sigma=0.1):
        self.weight = np.random.normal(mu, sigma, size=feature_size)
        self.bias = 0
        self.feature_size = feature_size
        self.lr = lr_init
        self.epoch = epoch

    def fit(self, feature, label):
        """train the model"""
        for e in xrange(self.epoch):
            for i in xrange(feature.shape[0]):
                if self.predict(np.reshape(feature[i, :], newshape=[1, feature.shape[1]])) != label[i]:
                    for j in xrange(self.feature_size):
                        self.weight[j] = self.weight[j] + self.lr * label[i] * feature[i][j]
                    self.bias = self.bias + self.lr * label[i]
            print("Epoch is %d, loss is %f" % (e, self.loss_rate(feature, label)))

    def predict(self, feature):
        """model inference process"""
        res = np.zeros((feature.shape[0]), dtype=np.float)
        # res = np.dot(feature, self.weight.transpose()) + self.bias
        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                res[i] = res[i] + feature[i][j] * self.weight[j]
            res[i] = res[i] + self.bias

        return self._activation(res)

    def _activation(self, data):
        data[data >= 0] = 1
        data[data < 0] = -1
        return data

    def loss_rate(self, feature, label):
        res = self.predict(feature)
        d = np.argwhere(res != label)
        return np.float(len(d)) / np.float(len(label))

    def __private_func(self):
        pass

    def _protected_func(self):
        pass


class PerceptronDual(Perceptron):
    """感知机的对偶情况，这里对标准表示进行继承，区别在于权重的更新公式，所以这里对权重的更新进行了重新设计"""
    def __init__(self,
                 feature_size,
                 lr_init,
                 epoch,
                 activation=None,
                 lr_decay_rate=None,
                 mu=0,
                 sigma=0.1):

        Perceptron.__init__(self,
                            feature_size,
                            lr_init,
                            epoch,
                            activation,
                            lr_decay_rate=None,
                            mu=0,
                            sigma=0.1)

        self.alpha = None

    def fit(self, feature, label):
        """train the model"""
        self.alpha = np.zeros(shape=len(label), dtype="float")
        # 初始化参数
        for i in range(feature.shape[0]):
            self.weight += self.alpha[i] * label[i] * feature[i]

        for e in range(self.epoch):
            for i in range(feature.shape[0]):
                if self.predict(np.reshape(feature[i, :], newshape=[1, feature.shape[1]])) != label[i]:
                    self.alpha[i] = self.alpha[i] + self.lr                  # 更新alpha
                    self.bias = self.bias + self.lr * label[i]               # 更新bias
                    for j in range(feature.shape[0]):
                        self.weight += self.alpha[j] * label[j] * feature[j, :]   # 是一个向量

            print("Epoch is %d, loss is %f" % (e, self.loss_rate(feature, label)))




