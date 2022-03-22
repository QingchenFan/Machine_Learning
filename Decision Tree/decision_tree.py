#coding:utf-8
'''
@author:fanqingchen
date:2022-03-21
Machine Learn：Decision Tree
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math
from math import log

import pprint

def create_data():
    datasets = [['青年', '否', '否', '一般', '否'],
               ['青年', '否', '否', '好', '否'],
               ['青年', '是', '否', '好', '是'],
               ['青年', '是', '是', '一般', '是'],
               ['青年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '一般', '否'],
               ['中年', '否', '否', '好', '否'],
               ['中年', '是', '是', '好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['中年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '非常好', '是'],
               ['老年', '否', '是', '好', '是'],
               ['老年', '是', '否', '好', '是'],
               ['老年', '是', '否', '非常好', '是'],
               ['老年', '否', '否', '一般', '否'],
               ]
    labels = [u'年龄', u'有工作', u'有自己的房子', u'信贷情况', u'类别']
    # 返回数据集和每个维度的名称
    return datasets, labels
dataset,labels = create_data()
train_data = pd.DataFrame(dataset,columns=labels)
#print(train_data)
d = {'青年':1, '中年':2, '老年':3, '一般':1, '好':2, '非常好':3, '是':0, '否':1}
#'青年1', '否1', '否1', '一般1', '否1'
data = []
for i in range(15):
    tmp = []
    t = dataset[i]
    for tt in t:            #tt 分别为'老年', '否', '否', '一般', '否'（举个例子）
        tmp.append(d[tt])   # d[tt] => d['青年']  print(d['青年']) =>1
    data.append(tmp)        #这样data列表里全为数字
data = np.array(data)       #将数字列表转换成矩阵形式

#Mark
print(data,data.shape)

X, y = data[:,:-1], data[:, -1] #X为训练集，y为标签

#Mark
print('label:',y)
print(set(y),type(y))
# 熵
def entropy(y):
    N = len(y)
    count = []
    for value in set(y):
        count.append(len(y[y == value]))
    count = np.array(count)
    #print('count:',count)
    entro = -np.sum((count / N) * (np.log2(count / N)))
    #print('entro:',entro)
    return entro

entropy(y)
print('熵:',entropy(y))

# 条件熵
def cond_entropy(X, y, cond):
    N = len(y)
    cond_X = X[:, cond]
    tmp_entro = []
    for val in set(cond_X):
        tmp_y = y[np.where(cond_X == val)]
        tmp_entro.append(len(tmp_y)/N * entropy(tmp_y))
    cond_entro = sum(tmp_entro)
    return cond_entro


cond_entropy(X, y, 0)

# 信息增益
def info_gain(X, y, cond):
    return entropy(y) - cond_entropy(X, y, cond)

# 信息增益比
def info_gain_ratio(X, y, cond):
    return (entropy(y) - cond_entropy(X, y, cond))/cond_entropy(X, y, cond)


# 信息增益
gain_a1 = info_gain(X, y, 0)

gain_a2 = info_gain(X, y, 1)

gain_a3 = info_gain(X, y, 2)

gain_a4 = info_gain(X, y, 3)


def best_split(X, y, method='info_gain'):
    """根据method指定的方法使用信息增益或信息增益比来计算各个维度的最大信息增益（比），返回特征的axis"""
    _, M = X.shape
    info_gains = []
    if method == 'info_gain':
        split = info_gain
    elif method == 'info_gain_ratio':
        split = info_gain_ratio
    else:
        print('No such method')
        return
    for i in range(M):
        tmp_gain = split(X, y, i)
        info_gains.append(tmp_gain)
    best_feature = np.argmax(info_gains)

    return best_feature

best_split(X,y)


def majorityCnt(y):
    """当特征使用完时，返回类别数最多的类别"""
    unique, counts = np.unique(y, return_counts=True)
    max_idx = np.argmax(counts)
    return unique[max_idx]

majorityCnt(y)


# #### ID3, C4.5算法

class DecisionTreeClassifer:
    """
    决策树生成算法，
    method指定ID3或C4.5,两方法唯一不同在于特征选择方法不同
    info_gain:       信息增益即ID3
    info_gain_ratio: 信息增益比即C4.5


    """

    def __init__(self, threshold, method='info_gain'):
        self.threshold = threshold
        self.method = method

    def fit(self, X, y, labels):
        labels = labels.copy()
        M, N = X.shape
        if len(np.unique(y)) == 1:
            return y[0]

        if N == 1:
            return majorityCnt(y)

        bestSplit = best_split(X, y, method=self.method)
        bestFeaLable = labels[bestSplit]
        Tree = {bestFeaLable: {}}
        del (labels[bestSplit])

        feaVals = np.unique(X[:, bestSplit])
        for val in feaVals:
            idx = np.where(X[:, bestSplit] == val)
            sub_X = X[idx]
            sub_y = y[idx]
            sub_labels = labels
            Tree[bestFeaLable][val] = self.fit(sub_X, sub_y, sub_labels)

        return Tree

My_Tree = DecisionTreeClassifer(threshold=0.1)
My_Tree.fit(X, y, labels)


# CART树
class CART:
    """CART树"""
    def __init__(self, ):
        "to be continue"


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    # print(data)
    return data[:,:2], data[:,-1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

clf = DecisionTreeClassifier()
clf.fit(data[:,:-1], data[:,-1])


clf.predict(np.array([1, 1, 0, 1]).reshape(1,-1)) # A

clf.predict(np.array([2, 0, 1, 2]).reshape(1,-1)) # B

clf.predict(np.array([2, 1, 0, 1]).reshape(1,-1)) # C


#tree_pic = export_graphviz(clf, out_file="mytree.pdf")
#with open('mytree.pdf') as f:
#    dot_graph = f.read()

#graphviz.Source(dot_graph)