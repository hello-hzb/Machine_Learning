#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
本测试代码参考scikit 分类demo plot_classifier_comparison.py进行设计，参考其测试数据对自己设计的高斯朴素贝叶斯算法进行测试
测试精度近似scikit设计的高斯朴素贝叶斯算法精度


作者：黄志彬
时间：20190811
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from x3_naive_bayes import GaussianNaiveBayes


h = .02  # step size in the mesh

names = ["Decision Tree", "Naive Bayes"]


X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), 2, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    # ---------------------核心代码测试-----------------------------------------
    clf = GaussianNaiveBayes(X_train, y_train)
    clf.fit()
    correct_num = 0
    for j in range(X_test.shape[0]):
        cls = clf.predict(X_test[j])
        if cls == y_test[j]:
            correct_num += 1
    print("Precision is ", float(correct_num)/float(len(y_test)))
    # -----------------------------------------------------------------------

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    # if hasattr(clf, "decision_function"):
    #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    # else:
    #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    #
    # # Put the result into a color plot
    # Z = Z.reshape(xx.shape)
    # ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
               edgecolors='k', alpha=0.6)

    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    if ds_cnt == 0:
        ax.set_title("Dataset")
    # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
    #         size=15, horizontalalignment='right')
    i += 1

plt.tight_layout()
plt.show()
