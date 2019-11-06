# -*- coding: utf-8 -*-
"""
# ######################################################################################
# 文件名称：x5_logistic_regression.py
# 摘   要：
# 作   者：hello-hzb
# 日   期：9/1/19
# 备   注：
# 算法知识点：
# 1.
# 2.
# python知识点：
# 1.
# 2.
# ######################################################################################
"""
import numpy as np
import matplotlib.pyplot as plt


class LogistcRegression(object):
    def __init__(self, epoch, batch_size, lr, lr_decay=None):
        self.weights = None
        self.lr = lr  # 学习率
        self.lr_decay = lr_decay
        self.training_epoch = epoch
        self.batch_size = batch_size

    def sigmoid(self, x):
        """
        sigmoid激活函数
        :param x: 待激活的输入值
        :return: 返回激活后数值
        """
        return 1.0 / (1.0 + np.exp(-x))

    def GD(self, datas, labels):
        """
        gradien descent,梯度下降算法，所有测试数据跑一遍，更新一次权重
        :param datas:
        :param labels:
        :return:
        """
        data_mat = np.mat(datas)  # convert to NumPy matrix
        label_mat = np.mat(labels).transpose()  # convert to NumPy matrix
        m, n = np.shape(data_mat)
        weights = np.ones((n, 1))
        for k in range(self.training_epoch):  # heavy on matrix operations
            y = self.sigmoid(data_mat * weights)  # matrix mult
            error = (label_mat - y)  # vector subtraction
            weights = weights + self.lr * data_mat.transpose() * error  # matrix mult
        return weights

    def BGD(self, datas, labels):
        """
        batch gradient descent,批次梯度下降算法，每次取一个批次的数据进行训练
        :param datas:
        :param labels:
        :return:
        """
        m, n = np.shape(datas)
        alpha = 0.01
        self.weights = np.ones(n)   # 初始化权重
        for i in range(m):          # batch_size为n
            h = self.sigmoid(np.sum(datas[i]*self.weights))
            error = labels[i] - h
            self.weights = self.weights + alpha * error * datas[i]

    def SGD(self, datas, labels):
        """
        stochastic gradient descent, 随机梯度下降算法，每次随机选择一个样本进行训练
        :param datas:
        :param labels:
        :return:
        """
        m, n = np.shape(datas)
        self.weights = np.ones(n)          # 初始化参数，最好改成随机数
        for j in range(self.training_epoch):
            # 每100轮训练，学习率衰减
            if j % 100 == 0:
                self.lr *= self.lr_decay
            datas_size = range(m)
            for i in range(m):
                rand_id = int(np.random.uniform(0, len(datas_size)))
                h = self.sigmoid(sum(datas[rand_id]*self.weights))
                error = labels[rand_id] - h
                self.weights = self.weights + self.lr * error * datas[rand_id]    # 参数更新公式
                del(datas_size[rand_id])

    def predict(self, x):
        prob = self.sigmoid(sum(x*self.weights))
        if prob > 0.5:
            return 1.0
        else:
            return 0.0


    def plotBestFit(self, dataMat, labelMat):
        dataArr = np.array(dataMat)
        n = np.shape(dataArr)[0]
        xcord1 = []
        ycord1 = []
        xcord2 = []
        ycord2 = []
        for i in range(n):
            if int(labelMat[i]) == 1:
                xcord1.append(dataArr[i, 1])
                ycord1.append(dataArr[i, 2])
            else:
                xcord2.append(dataArr[i, 1])
                ycord2.append(dataArr[i, 2])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
        ax.scatter(xcord2, ycord2, s=30, c='green')
        x = np.arange(-3.0, 3.0, 0.1)
        y = (-self.weights[0] - self.weights[1] * x) / self.weights[2]
        ax.plot(x, y)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10
    errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum/float(numTests))

