#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
算法知识点：
    1.距离的度量：欧式距离，曼哈顿距离，minkowski 距离等等
    4.k值如何选择：一般取较小的值，通过   交叉验证  的方式来选取

    1.KD树的作用：减少搜索最近k个样本点的计算开销，不需要计算和所有样本点的距离
    2.KD树的构建：将数据集的样本点根据每个特征的取值进行划分，以中值为节点，生成一个树型结构
    3.KD树的检索：
        （1）在kd树中找到包含目标节点x的叶节点：从根节点出发，递归向下访问kd树，若目标点x当前维的坐标小于切分点的坐标，则移动到左子节点，
        否则移动到右子节点，直到子节点为叶节点为止；
        （2）以此叶节点为“当前最近点”
        （3）递归向上回退，在每个节点进行以下操作：
            （a）如果该节点保存的样本点比当前最近距离目标点更近，则更新最近点为当前节点；
            （b）检查该子节点的父节点的另一个子节点是否包含更近的点。具体地，检查另一子节点对应的区域是否以目标点为球心，以当前最近距离为半径的超球体相交。
            若相交，可能在另外一个子节点对应的区域内存在和目标点x距离更近的点，移动到另外的一个子节点，递归进行最近邻搜索。
            若不相交，向上回退。
        （4）当回退到根节点的时候，搜索结束，返回最近点。

        K近邻需要搜索k个距离较小的样本点，这里只要声明一个能够存下k个样本点的即各个样本点与目标点x的距离的list，list的最后一个元素记录距离最大的样本点，
        每次遍历过一个节点，计算该节点与目标点的距离，并与已经记录的k个最近样本点的最大距离进行比较，若当前节点的距离小于已经记录的最大距离，
        将该最大距离的替换为当前节点，并进行排序，确保list最后一个元素是离目标点距离最大的点。


python知识点：
    1.对多维数值按照某一个维度的数据进行排序  sort
    2.统计一个数组中出现次数最多的元素np.argmax(np.bincount()),类似哈希表

"""
import numpy as np
# https://github.com/stefankoegl/kdtree/blob/master/kdtree.py
# http://www.voidcn.com/article/p-zmvbhjlz-qq.html
# https://www.cnblogs.com/21207-iHome/p/6084670.html


class KNeighborClassifier(object):
    def __init__(self,
                 dataset,
                 labels,
                 metric='minkowski'):
        self.kd_tree = KdTree(dataset, labels)

    def predict(self, feature, k):
        """feature shape is (n_instance, n_feature)"""
        closestpoints = np.zeros((k, len(feature)+2), dtype=type(feature))
        self.kd_tree.findKNode(self.kd_tree.root, closestpoints, feature, k)
        # np.bincount 返回一个数组，每个元素代表该id出现的次数，如生成一个7元素数组，其中索引3位置的元素表示3出现的次数
        # np.argmax 获得数组元素中最大值的索引，所以和np.bincount结合起来用就能够获得出现次数最大的类别了
        # https://blog.csdn.net/y12345678904/article/details/72852174
        # 有点像哈希表，数组的id表示统计的而元素，id对应的数值表示该元素出现的次数
        return np.argmax(np.bincount(closestpoints[:, -1]))


class KdNode(object):
    """kd树中的每个节点的数据结构"""
    def __init__(self, node_data, split_axis, left, right):
        self.node_data = node_data      # 落在当前切割超平面上的样本点
        self.split_axis = split_axis    # 当前切割超平面对应的特征的维度
        self.left = left                # 该结点分割超平面左子空间构成的kd子树的根节点
        self.right = right              # 该结点分割超平面右子空间构成的kd子数的根节点
        # self.classification


class KdTree(object):
    """kd树的创建，data为list"""
    def __init__(self, datas, labels):
        k = len(data[0])                  # 特征维度
        dataset = np.column_stack(datas, labels)

        def CreateNode(split, data_set):  # 按样本第split维特征进行数据集划分创建KdNode
            if not data_set:              # 数据集为空时，一种情况最开始的时候训练数据集为空，另外一种情况就是kd树创建完成返回
                return None
            data_set = data_set[data_set[:, split].argsort()]
            # data_set.sort(key=lambda x: x[split])   # 按照split轴的数据进行排序
            split_pos = len(data_set) // 2          # 获得中位数的位置
            median = data_set[split_pos]            # 中位数分割点
            split_next = (split + 1) % k            # 更新切分轴

            # 递归的创建kd树
            return KdNode(median, split,
                          CreateNode(split_next, data_set[:split_pos]),      # 创建左子树
                          CreateNode(split_next, data_set[split_pos + 1:]))  # 创建右子树

        self.root = CreateNode(0, dataset)  # 从第0维分量开始构建kd树,返回根节点
        self.closest_point = None
        self.min_dist = -1

        # https://segmentfault.com/a/1190000016293317
    def findClosest(self, kdNode, closestPoint, x, minDis):
        """
        这里存在一个问题，当传递普通的不可变对象minDis时，递归退回第一次找到
        最端距离前，minDis改变，最后结果混乱，这里传递一个可变对象进来。
        kdNode:是构造好的kd树。
        closestPoint：是存储最近点的可变对象，这里是array
        x：是要预测的实例
        minDis：是当前最近距离。
        """
        if kdNode == None:
            return
        # 计算欧氏距离
        curDis = (sum((kdNode.value[0:-2] - x[0:-2])**2))**0.5
        if minDis < 0 or curDis < minDis:
            minDis = curDis
            closestPoint = kdNode.value
        # 递归查找叶节点
        if kdNode.value[kdNode.dimension] >= x[kdNode.dimension]:
            self.findClosest(kdNode.left, closestPoint, x, minDis)
        else:
            self.findClosest(kdNode.right, closestPoint, x, minDis)
        # 计算测试点和分隔超平面的距离，如果相交进入另一个叶节点重复
        rang = abs(x[kdNode.dimension] - kdNode.value[kdNode.dimension])
        if rang > minDis:
            return
        if kdNode.value[kdNode.dimension] >= x[kdNode.dimension]:
            self.findClosest(kdNode.right, closestPoint, x, minDis)   # 往另外一侧的节点搜索，和上面递归的方向是相反的
        else:
            self.findClosest(kdNode.left, closestPoint, x, minDis)

    def findKNode(self, kdNode, closestPoints, x, k):
        """
        k近邻搜索，kdNode是要搜索的kd树
        closestPoints:是要搜索的k近邻点集合,将minDis放入closestPoints最后一列合并
        x：预测实例
        minDis：是最近距离
        k:是选择k个近邻
        """
        if kdNode == None:
            return
        # 计算欧式距离
        curDis = (sum((kdNode.value[0:-2] - x) ** 2)) ** 0.5
        # 将closestPoints按照minDis列排序,这里存在一个问题，排序后返回一个新对象
        closestPoints = closestPoints[closestPoints[:, -1].argsort()]
        # 每次取最后一行元素操作, 即和当前最大的距离作比较，确定是否更新最近的k个点，每更新一次会做一次排序
        if closestPoints[-1][-1] >= 10000 or closestPoints[-1][-1] > curDis:
            closestPoints[-1][-1] = curDis
            closestPoints[-1, 0:-2] = kdNode.value

            # 递归搜索叶结点
        if kdNode.value[kdNode.dimension] >= x[kdNode.dimension]:
            self.findKNode(kdNode.left, closestPoints, x, k)
        else:
            self.findKNode(kdNode.right, closestPoints, x, k)
        # 计算测试点和分隔超平面的距离，如果相交进入另一个叶节点重复
        rang = abs(x[kdNode.dimension] - kdNode.value[kdNode.dimension])
        if rang > closestPoints[k - 1][-1]:
            return
        if kdNode.value[kdNode.dimension] >= x[kdNode.dimension]:
            self.findKNode(kdNode.right, closestPoints, x, k)
        else:
            self.findKNode(kdNode.left, closestPoints, x, k)


if __name__ == "__main__":
    data = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    kd = KdTree(data)
    preorder(kd.root)