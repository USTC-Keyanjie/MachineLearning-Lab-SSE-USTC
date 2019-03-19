# -*- coding: utf-8 -*-
# @File     : decision_tree_iris.py
# @Author   : Yanjie Ke
# @Date     : 2019/1/6 10:54
# @License  : Copyright(C), USTC

import numpy as np
from decision_tree import *
from paint_tree import *
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random

def pre_processing_data():
    # 得到数据集
    iris = datasets.load_iris()
    iris_feature, iris_target = iris.data, iris.target
    iris_data = []
    # print(iris_feature)

    # 将特征和label连接在一起，转换成列表
    for i in range(len(iris_feature)):
        sample = list(iris_feature[i])
        sample.append(iris_target[i])
        iris_data.append(sample)

    # 第一个特征中最大值
    sepal_length_max = max(iris_feature.transpose()[0])
    # 第一个特征中最小值
    sepal_length_min = min(iris_feature.transpose()[0])
    sepal_width_max = max(iris_feature.transpose()[1])
    sepal_width_min = min(iris_feature.transpose()[1])
    petal_length_max = max(iris_feature.transpose()[2])
    petal_length_min = min(iris_feature.transpose()[2])
    petal_width_max = max(iris_feature.transpose()[3])
    petal_width_min = min(iris_feature.transpose()[3])

    processed_feature = []
    for i in range(len(iris_data)):
        sepal_length, sepal_width, petal_length, petal_width, target = iris_data[i]

        target = int(target)

        # 这些是划分类别的，均分4份，划成4类
        if sepal_length < sepal_length_min + (sepal_length_max - sepal_length_min) / 4:
            sepal_length = 1
        elif sepal_length >= sepal_length_min + (
                sepal_length_max - sepal_length_min) / 4 and sepal_length < sepal_length_min + 2 * (
                sepal_length_max - sepal_length_min) / 4:
            sepal_length = 2
        elif sepal_length >= sepal_length_min + 2 * (
                sepal_length_max - sepal_length_min) / 4 and sepal_length < sepal_length_min + 3 * (
                sepal_length_max - sepal_length_min) / 4:
            sepal_length = 3
        elif sepal_length >= sepal_length_min + 3 * (sepal_length_max - sepal_length_min) / 4:
            sepal_length = 4

        # sepal_width
        if sepal_width < sepal_width_min + (sepal_width_max - sepal_width_min) / 4:
            sepal_width = 1
        elif sepal_width >= sepal_width_min + (
                sepal_width_max - sepal_width_min) / 4 and sepal_width < sepal_width_min + 2 * (
                sepal_width_max - sepal_width_min) / 4:
            sepal_width = 2
        elif sepal_width >= sepal_width_min + 2 * (
                sepal_width_max - sepal_width_min) / 4 and sepal_width < sepal_width_min + 3 * (
                sepal_width_max - sepal_width_min) / 4:
            sepal_width = 3
        elif sepal_width >= sepal_width_min + 3 * (sepal_width_max - sepal_width_min) / 4:
            sepal_width = 4

        # petal_length
        if petal_length < petal_length_min + (petal_length_max - petal_length_min) / 4:
            petal_length = 1
        elif petal_length >= petal_length_min + (
                petal_length_max - petal_length_min) / 4 and petal_length < petal_length_min + 2 * (
                petal_length_max - petal_length_min) / 4:
            petal_length = 2
        elif petal_length >= petal_length_min + 2 * (
                petal_length_max - petal_length_min) / 4 and petal_length < petal_length_min + 3 * (
                petal_length_max - petal_length_min) / 4:
            petal_length = 3
        elif petal_length >= petal_length_min + 3 * (petal_length_max - petal_length_min) / 4:
            petal_length = 4

        # petal_width
        if petal_width < petal_width_min + (petal_width_max - petal_width_min) / 4:
            petal_width = 1
        elif petal_width >= petal_width_min + (
                petal_width_max - petal_width_min) / 4 and petal_width < petal_width_min + 2 * (
                petal_width_max - petal_width_min) / 4:
            petal_width = 2
        elif petal_width >= petal_width_min + 2 * (
                petal_width_max - petal_width_min) / 4 and petal_width < petal_width_min + 3 * (
                petal_width_max - petal_width_min) / 4:
            petal_width = 3
        elif petal_width >= petal_width_min + 3 * (petal_width_max - petal_width_min) / 4:
            petal_width = 4
        processed_feature.append([sepal_length, sepal_width, petal_length, petal_width, target])

    return processed_feature


# 划分数据集
def split_dataset(dataset, splitRatio=0.67):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return trainSet, copy  # 返回训练集与测试集


if __name__ == '__main__':
    # 得到处理过的数据
    data = pre_processing_data()
    # print('len_data :', data)

    # 划分数据集
    train_set, test_set = split_dataset(data)
    feature = ['sepal length', 'sepal width', 'petal length', 'petal width']

    # 构造决策树
    iris_tree = createTree(train_set, feature, criterion='ID3')
    print(iris_tree)

    # 检查正确率
    acc_count = 0
    feature = ['sepal length', 'sepal width', 'petal length', 'petal width']
    for i in range(len(test_set)):
        prediction = classify(iris_tree, feature, test_set[i][:-1])  # label不要传进去了
        if prediction == test_set[i][-1]:
            acc_count += 1
    accuracy = acc_count / len(test_set)
    print('accuracy = ', accuracy)

    # 画出这棵树
    createPlot(iris_tree)