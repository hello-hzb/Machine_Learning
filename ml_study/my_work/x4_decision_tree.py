#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
https://blog.csdn.net/leaf_zizi/article/details/83105836
https://www.cnblogs.com/sumuncle/p/5760458.html  矩阵操作
获得矩阵的维度，形状，元素个数

print('number of dim:',array.ndim)
print('shape:', array.shape)
print('size:', array.size)

https://www.cnblogs.com/xn5991/p/9526267.html
按照条件返回ndarray中的数据，返回一个子矩阵

"""
import numpy as np
from math import log
import operator
import x4_treePlotter


class DecisionTree(object):
    """
    决策树
    """
    def __init__(self,
                 dataset,              # n * feature_size
                 labels=None,          # n
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 ):
        # self.dataset = np.column_stack((datas, labels))
        self.dataset = dataset
        self.tree = None

    def __information_gain_ratio(self, dataset):
        """信息增益比的计算"""
        feature_size = dataset.shape[1] - 1    # 特征个数
        sample_num = dataset.shape[0]          # 样本个数
        # 计算经验熵
        cls_set = np.unique(dataset[:, -1])
        entropy_tmp = 0.0
        for j in range(cls_set.size):
            tmp = (dataset[:, -1] == cls_set[j]).sum()
            prob = float(tmp) / sample_num
            entropy_tmp -= prob * log(prob)
        empirical_entropy = entropy_tmp     # 经验熵

        # 计算条件经验熵
        info_gain_ratio_list = []
        # 1.遍历特征的循环
        for i in range(feature_size):
            feat_value_set = np.unique(dataset[:, i])       # 特征的取值集合
            # global_prob = 0.0
            empirical_condition_entropy_tmp = 0.0           # 经验条件熵H(D|A)
            feat_entropy = 0.0
            # 2.遍历特征i所有取值的循环
            for j in range(feat_value_set.size):
                # 取出特征i取值为feat_value_set[j]的样本
                tmp_data = dataset[np.where(dataset[:, i] == feat_value_set[j])]
                global_prob = float(tmp_data.shape[0]) / sample_num  # Di/D
                feat_entropy -= global_prob * log(global_prob, 2)    # 特征的熵HA(D)
                # 获得子样本内的类别
                local_cls_set = np.unique(tmp_data[:, -1])
                condition_entropy_tmp = 0.0                 # 条件熵
                # 3.遍历特征i取值j的样本类别的循环
                for k in range(local_cls_set.size):
                    tmp = (tmp_data[:, -1] == local_cls_set[k]).sum()
                    local_prob = float(tmp) / tmp_data.shape[0]
                    condition_entropy_tmp += local_prob * log(local_prob)

                empirical_condition_entropy_tmp -= global_prob * condition_entropy_tmp
            # 计算信息增益比 g(D, A) / HA(D)
            info_gain_ratio = (empirical_entropy - empirical_condition_entropy_tmp) / feat_entropy
            info_gain_ratio_list.append(info_gain_ratio)
        return info_gain_ratio_list

    def __split_dataset(self, dataset, axis, value):
        """
        输入：数据集，选择维度，选择值
        输出：划分数据集
        描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
        """

        tmp_data = dataset[np.where(dataset[:, axis] == value)]
        sub_dataset = np.delete(tmp_data, axis, axis=1)
        return sub_dataset

    def __feature_select(self, dataset):
        """
        输入：数据集
        输出：最好的划分维度
        描述：选择最好的数据集划分维度
        """
        info_gain_ratio_list = self.__information_gain_ratio(dataset)
        return np.argmax(np.array(info_gain_ratio_list))

    def __majorityCnt(self, cls_list):
        """
        输入：分类类别列表
        输出：子节点的分类
        描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
              采用多数判决的方法决定该子节点的分类
        """
        cls_cnt = {}
        for vote in cls_list:
            if vote not in cls_cnt.keys():
                cls_cnt[vote] = 0
            cls_cnt[vote] += 1
        sortedClassCount = sorted(cls_cnt.iteritems(), key=operator.itemgetter(1), reversed=True)
        return sortedClassCount[0][0]

    def __fit(self, dataset, labels):
        """
        输入：数据集，特征标签
        输出：决策树
        描述：递归构建决策树，利用上述的函数
        """
        # cls_list = [example[-1] for example in dataset]
        cls_list = list(dataset[:, -1])
        if cls_list.count(cls_list[0]) == len(cls_list):
            # 类别完全相同，停止划分
            return cls_list[0]
        if len(dataset[0]) == 1:
            # 遍历完所有特征时返回出现次数最多的
            return self.__majorityCnt(cls_list)
        best_feat_id = self.__feature_select(dataset)
        best_feat_label = labels[best_feat_id]     # 信息增益比最大的特征
        mytree = {best_feat_label: {}}
        del(labels[best_feat_id])
        # 得到列表包括节点所有的属性值
        feat_values = dataset[:, best_feat_id]
        unique_vals = np.unique(feat_values)
        for value in unique_vals:
            sub_labels = labels[:]
            sub_dataset = self.__split_dataset(dataset, best_feat_id, value)
            mytree[best_feat_label][value] = self.__fit(sub_dataset, sub_labels)
        return mytree

    def __classify(self, tree, feat_labels, testVec):
        """
        输入：决策树，分类标签，测试数据
        输出：决策结果
        描述：跑决策树
        """

        cls_label = None
        first_str = list(tree.keys())[0]
        second_dict = tree[first_str]
        feat_id = feat_labels.index(first_str)
        for key in second_dict.keys():
            if testVec[feat_id] == key:     # 判断当前测试数据进入哪个分支
                if type(second_dict[key]).__name__ == 'dict':    # 如果该节点还有子节点，需要进一步判断类别
                    cls_label = self.__classify(second_dict[key], feat_labels, testVec)
                else:
                    cls_label = second_dict[key]
                    break
        return cls_label

    def fit(self, dataset, labels):
        self.tree = self.__fit(dataset, labels)

    def predict(self, feat_labels, test_dataset):
        """
        输入：决策树，分类标签，测试数据集
        输出：决策结果
        描述：跑决策树
        """

        predict_res = []
        for testVec in test_dataset:
            predict_res.append(self.__classify(feat_labels, testVec))
        return predict_res

    def tree_pruning(self):
        """
        决策树剪枝
        :return:
        """
        pass

    def store_tree(self, filename):
        """
        输入：决策树，保存文件路径
        输出：
        描述：保存决策树到文件
        """
        import pickle
        fw = open(filename, 'wb')
        pickle.dump(self.tree, fw)
        fw.close()

    def grab_tree(self, filename):
        """
        输入：文件路径名
        输出：决策树
        描述：从文件读取决策树
        """
        import pickle
        fr = open(filename, 'rb')
        self.tree = pickle.load(fr)


class CART(DecisionTree):
    """

    """
    def __init__(self):
        pass



def createDataSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    dataSet = [[0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0],
               [1, 0, 0, 0, 1],
               [2, 1, 0, 0, 1],
               [2, 2, 1, 0, 1],
               [2, 2, 1, 1, 0],
               [1, 2, 1, 1, 1]]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels


def createTestSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true
    """
    testSet = [[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    return testSet


def main():
    dataset, labels = createDataSet()
    labels_tmp = labels[:]          # 拷贝，createTree会改变labels

    tree = DecisionTree(dataset)
    tree.fit(np.array(dataset), labels_tmp)
    # tree.store_tree('classifierStorage.txt')     # 保存决策树
    # tree.grab_tree('classifierStorage.txt')      # 加载决策树
    print('desicionTree:\n', tree.tree)
    x4_treePlotter.createPlot(tree.tree)
    testSet = createTestSet()
    print('classifyResult:\n', tree.predict(labels, np.array(testSet)))


if __name__ == '__main__':
    main()







# def calcShannonEnt(dataSet):
#     """
#     输入：数据集
#     输出：数据集的香农熵
#     描述：计算给定数据集的香农熵；熵越大，数据集的混乱程度越大，又称为经验熵
#     """
#     numEntries = len(dataSet)
#     labelCounts = {}
#     # 获得各个标签对应的样本个数
#     for featVec in dataSet:
#         currentLabel = featVec[-1]
#         if currentLabel not in labelCounts.keys():
#             labelCounts[currentLabel] = 0
#         labelCounts[currentLabel] += 1
#     shannonEnt = 0.0
#     for key in labelCounts:
#         prob = float(labelCounts[key])/numEntries
#         shannonEnt -= prob * log(prob, 2)
#     return shannonEnt
#
#
# def splitDataSet(dataSet, axis, value):
#     """
#     输入：数据集，选择维度，选择值
#     输出：划分数据集
#     描述：按照给定特征划分数据集；去除选择维度中等于选择值的项
#     """
#     retDataSet = []
#     for featVec in dataSet:
#         if featVec[axis] == value:
#             reduceFeatVec = featVec[:axis]
#             reduceFeatVec.extend(featVec[axis+1:])
#             retDataSet.append(reduceFeatVec)
#     return retDataSet
#
#
# def chooseBestFeatureToSplit(dataSet):
#     """
#     输入：数据集
#     输出：最好的划分维度
#     描述：选择最好的数据集划分维度
#     """
#     numFeatures = len(dataSet[0]) - 1
#     baseEntropy = calcShannonEnt(dataSet)       # 计算经验熵
#     bestInfoGainRatio = 0.0
#     bestFeature = -1
#     for i in range(numFeatures):
#         featList = [example[i] for example in dataSet]
#         uniqueVals = set(featList)
#         newEntropy = 0.0      # 经验条件熵
#         splitInfo = 0.0
#         for value in uniqueVals:
#             subDataSet = splitDataSet(dataSet, i, value)
#             prob = len(subDataSet)/float(len(dataSet))
#             newEntropy += prob * calcShannonEnt(subDataSet)
#             splitInfo += -prob * log(prob, 2)
#         infoGain = baseEntropy - newEntropy      # 信息增益
#         if (splitInfo == 0): # fix the overflow bug
#             continue
#         infoGainRatio = infoGain / splitInfo     # 信息增益比
#         if (infoGainRatio > bestInfoGainRatio):
#             bestInfoGainRatio = infoGainRatio
#             bestFeature = i
#     return bestFeature
#
#
# def majorityCnt(classList):
#     """
#     输入：分类类别列表
#     输出：子节点的分类
#     描述：数据集已经处理了所有属性，但是类标签依然不是唯一的，
#           采用多数判决的方法决定该子节点的分类
#     """
#     classCount = {}
#     for vote in classList:
#         if vote not in classCount.keys():
#             classCount[vote] = 0
#         classCount[vote] += 1
#     sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reversed=True)
#     return sortedClassCount[0][0]
#
#
# def createTree(dataSet, labels):
#     """
#     输入：数据集，特征标签
#     输出：决策树
#     描述：递归构建决策树，利用上述的函数
#     """
#     classList = [example[-1] for example in dataSet]
#     if classList.count(classList[0]) == len(classList):
#         # 类别完全相同，停止划分
#         return classList[0]
#     if len(dataSet[0]) == 1:
#         # 遍历完所有特征时返回出现次数最多的
#         return majorityCnt(classList)
#     bestFeat = chooseBestFeatureToSplit(dataSet)
#     bestFeatLabel = labels[bestFeat]     # 信息增益比最大的特征
#     myTree = {bestFeatLabel:{}}
#     del(labels[bestFeat])
#     # 得到列表包括节点所有的属性值
#     featValues = [example[bestFeat] for example in dataSet]
#     uniqueVals = set(featValues)
#     for value in uniqueVals:
#         subLabels = labels[:]
#         myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
#     return myTree
#
#
# def classify(inputTree, featLabels, testVec):
#     """
#     输入：决策树，分类标签，测试数据
#     输出：决策结果
#     描述：跑决策树
#     """
#     firstStr = list(inputTree.keys())[0]
#     secondDict = inputTree[firstStr]
#     featIndex = featLabels.index(firstStr)
#     for key in secondDict.keys():
#         if testVec[featIndex] == key:     # 判断当前测试数据进入哪个分支
#             if type(secondDict[key]).__name__ == 'dict':    # 如果该节点还有子节点，需要进一步判断类别
#                 classLabel = classify(secondDict[key], featLabels, testVec)
#             else:
#                 classLabel = secondDict[key]
#                 break
#     return classLabel
#
#
# def classifyAll(inputTree, featLabels, testDataSet):
#     """
#     输入：决策树，分类标签，测试数据集
#     输出：决策结果
#     描述：跑决策树
#     """
#     classLabelAll = []
#     for testVec in testDataSet:
#         classLabelAll.append(classify(inputTree, featLabels, testVec))
#     return classLabelAll
#
#
# def storeTree(inputTree, filename):
#     """
#     输入：决策树，保存文件路径
#     输出：
#     描述：保存决策树到文件
#     """
#     import pickle
#     fw = open(filename, 'wb')
#     pickle.dump(inputTree, fw)
#     fw.close()
#
#
# def grabTree(filename):
#     """
#     输入：文件路径名
#     输出：决策树
#     描述：从文件读取决策树
#     """
#     import pickle
#     fr = open(filename, 'rb')
#     return pickle.load(fr)
