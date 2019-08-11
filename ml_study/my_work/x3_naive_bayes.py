#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
朴素贝叶斯法算法知识点：
    1.离散特征值：
    先验概率计算和后验概率计算通过对样本进行计数的方式进行计算, 计算方式又分为极大似然估计和贝叶斯估计，其中：
    贝叶斯估计在计算先验概率和后验概率的时候会在分子分母位置加上系数，这样可以避免由于样本不足的时候，概率计算为0的情况。


    2.连续特征值：
    计算每个类别的每个特征的均值与方差，采用高斯概率密度函数计算后验概率

    3.该算法有个前提，即各个特征互相独立



python知识点：
    1.返回数组中出现的元素及其出现的次数：
        # 生成一组int测试数据
        a = np.random.randint(-1, 1, size=(10, 10))
        # 返回数组中出现的数值及其出现的次数
        u, indices = np.unique(a[:, 1], return_counts=True)
        b = a[a[:, 9] == -1]
        # 返回某一个数组中某个元素出现的次数
        print(np.sum(a[:, 9] == -1))
        # 返回数组中出现的数值及其出现的次数
        a = np.array([1, 2, 6, 4, 2, 3, 2])
        u, indices = np.unique(a, return_counts=True)


    2.ndarray与list互相转换
        a = np.ones((10, 10)
        b = [1, 1, 1, 1, 1]

        b_ndarray = np.array(b)
        a_list = a.tolist()
"""
import numpy as np
import math


class NaiveBayes(object):
    """朴素贝叶斯法，离散特征值的分类，未做测试"""
    def __init__(self,
                 dataset,              # 数据集
                 label,                # 数据集的标签
                 feature_value_set,    # 每个特征值的取值范围，一行代表一个特征，多个特征多行，且每个特征的取值范围个数相同
                                       # 后续对后延概率计算做修改，切换成list，这样就不用要求每个特征的取值范围个数一致
                 label_value_set,      # 类别的集合，即有那些类别，不是数据集的类别
                 ):
        self.dataset = dataset
        self.label = label
        self.feature_value_set = feature_value_set
        self.label_value_set = label_value_set
        self.prior_prob_dict = []
        self.after_prob_dict = {}

    def fit(self):
        """模型参数计算"""
        # 贝叶斯估计
        self.__bayesian_estimation(lamb=1)
        # 极大似然估计
        self.__mle()

    def predict(self, x):
        """推理"""
        res = []
        temp = 1.0
        for label in self.label_value_set:
            for i in range(x.shape[0]):
                idx = np.where(self.feature_value_set[i, :] == x[i])
                temp *= self.after_prob_dict[label][i, idx[0]]
            idx = np.where(self.label_value_set == label)
            temp *= self.prior_prob_dict[idx[0]]
            res.append(temp)

        return self.label_value_set[np.argmax(np.array(res))]

    def __bayesian_estimation(self, lamb):
        """贝叶斯估计"""
        """每个标签出现的次数"""
        label_count = np.bincount(self.label)
        for i in self.label_value_set:
            self.prior_prob_dict.append(label_count[i])

        """后验概率计算，计算标签发生的情况下, 特征值为某个值的发生概率"""
        data_stack = np.column_stack(self.dataset, self.label)

        # 一行代表每个特征的不同取值的概率，每一行代表一个特征
        temp = np.zeros(shape=self.feature_value_set.shape)
        for i in self.label_value_set:
            # 获得某一个类别的所有样本点
            sub_dataset = data_stack[data_stack[:, -1] == i]
            for j in range(self.feature_value_set.shape[0]):
                for k in range(self.feature_value_set.shape[1]):
                    # 统计第j个特征的第k个取值的样本个数，并除于该类别的样本个数
                    temp[j, k] = (lamb + np.sum(sub_dataset[:, j] == self.feature_value_set[j, k]))\
                                 / (self.prior_prob_dict[i] + lamb * self.feature_value_set.shape[1])
            self.after_prob_dict[i] = temp

        """计算每个类别的先验概率"""
        dataset_size = len(self.label)
        label_value_size = len(self.label_value_set)
        # 计算每个类别的先验概率
        self.prior_prob_dict = (self.prior_prob_dict + lamb) / (dataset_size + lamb * label_value_size)

    def __mle(self):
        """极大似然估计方法（Maximum Likelihood Estimate，MLE）"""
        """每个标签出现的次数"""
        label_count = np.bincount(self.label)
        for i in self.label_value_set:
            self.prior_prob_dict.append(label_count[i])

        """后验概率计算，计算标签发生的情况下, 特征值为某个值的发生概率"""
        data_stack = np.column_stack(self.dataset, self.label)

        # 一行代表每个特征的不同取值的概率，每一行代表一个特征
        temp = np.zeros(shape=self.feature_value_set.shape)
        for i in self.label_value_set:
            # 获得某一个类别的所有样本点
            sub_dataset = data_stack[data_stack[:, -1] == i]
            for j in range(self.feature_value_set.shape[0]):
                for k in range(self.feature_value_set.shape[1]):
                    # 统计第j个特征的第k个取值的样本个数
                    temp[j, k] = np.sum(sub_dataset[:, j] == self.feature_value_set[j, k]) / self.prior_prob_dict[i]
            self.after_prob_dict[i] = temp

        """计算每个类别的先验概率"""
        dataset_size = len(self.label)
        # 计算每个类别的先验概率
        self.prior_prob_dict = self.prior_prob_dict / dataset_size

    def __prior_probability_cal(self):
        """计算先验概率"""
        # 实现方式1,将ndarray转成list后，使用count的方式获得每个标签出现的次数
        # label.tolist()
        # label_dict = {}
        # for i in self.label_set:
        #     label_dict[i] = label.count(i)

        # 实现方式2，对ndarray直接计算，bincount获得一个数组，id为标签值，元素值为标签值出现的次数
        label_count = np.bincount(self.label)
        for i in self.label_value_set:
            self.prior_prob_dict.append(label_count[i])

    def __condition_probability_cal(self):
        """计算标签发生的情况下, 特征值为某个值的发生概率"""
        data_stack = np.column_stack(self.dataset, self.label)

        # 一行代表每个特征的不同取值的概率，每一行代表一个特征
        temp = np.zeros(shape=self.feature_value_set.shape)
        for i in self.label_value_set:
            # 获得某一个类别的所有样本点
            sub_dataset = data_stack[data_stack[:, -1] == i]
            for j in range(self.feature_value_set.shape[0]):
                for k in range(self.feature_value_set.shape[1]):
                    # 统计第j个特征的第k个取值的样本个数
                    temp[j, k] = np.sum(sub_dataset[:, j] == self.feature_value_set[j, k]) / self.prior_prob_dict[i]
            self.after_prob_dict[i] = temp


class GaussianNaiveBayes(object):
    """高斯朴素贝叶斯法，用于处理特征值为连续的样本"""
    def __init__(self,
                 dataset,
                 label,
                 ):
        self.dataset = dataset                    # 数据集
        self.label = label                        # 数据集的标签
        self.label_value_set = list(set(label))   # 类别集合
        self.prior_prob_dict = []                 # 存储先验概率
        self.after_prob_dict = {}                 # 存储后验概率

    def fit(self):
        """模型训练，朴素贝叶斯算法的训练过程实质上就是获得先验概率和后验概率"""
        # 极大似然估计
        self.__mle()

    def predict(self, x):
        """推理"""
        res = []
        for label in self.label_value_set:
            temp = 1.0
            for i in range(x.shape[0]):                       # 第i个特征
                mu = self.after_prob_dict[label][i, 0]        # 每个类别样本中每个特征的均值
                theta = self.after_prob_dict[label][i, 1]     # 每个类别样本中每个特征的方差
                # 每个特征的后验概率计算替换为高斯概率密度函数
                temp *= (1/((2 * math.pi * theta)**0.5)) * np.exp(-(((x[i]-mu)**2) / (2 * theta)))
            idx = np.where(self.label_value_set == label)

            temp *= self.prior_prob_dict[int(idx[0])]         # 后验概率乘于先验概率得到测试样本分类为label的概率
            res.append(temp)
        max_label_id = np.argmax(np.array(res))

        return self.label_value_set[int(max_label_id)]        # 返回概率最大的label

    def __mle(self):
        """极大似然估计方法（Maximum Likelihood Estimate，MLE）"""
        """计算每个类别的先验概率"""
        # 数据集中每个类别样本个数
        label_count = np.bincount(self.label)
        dataset_size = float(len(self.label))
        for i in self.label_value_set:
            self.prior_prob_dict.append(float(label_count[i]) / dataset_size)

        """后验概率计算，计算每个类别中, 特征值为某个值的发生概率"""
        data_stack = np.column_stack((self.dataset, self.label))   # 将数据集和标签绑定到一起

        # 每一行代表一个特征, 每个特征两个数据，均值与方差
        for i in self.label_value_set:

            sub_dataset = data_stack[data_stack[:, -1] == i]       # 获得某一个类别的所有样本点
            # 分别计算每个特征的均值与方差
            self.after_prob_dict[i] = np.zeros(shape=(self.dataset.shape[1], 2))
            for j in range(self.after_prob_dict[i].shape[0]):
                self.after_prob_dict[i][j, 0] = np.mean(sub_dataset[:, j])  # 计算第j个特征的均值
                self.after_prob_dict[i][j, 1] = np.var(sub_dataset[:, j])   # 计算第j个特征的方差


#
# a = np.array([1, 2, 6, 4, 2, 3, 2])
# u = np.where(a == 2)
# print("jj")
#
# a = []
# b = [1, 2, 3]
# c = [1, 2, 4]
# d = [1, 6, 3]
#
# a.append(b)
# a.append(c)
# a.append(d)
#
# print(a[:, 2])
