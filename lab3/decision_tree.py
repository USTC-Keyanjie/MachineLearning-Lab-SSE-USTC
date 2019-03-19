# -*- coding: utf-8 -*-
# @File     : decision_tree.py
# @Author   : Yanjie Ke
# @Date     : 2019/1/6 10:16
# @License  : Copyright(C), USTC
from math import log
import operator
import pickle


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    feature = ['no surfacing', 'flippers']
    return dataSet, feature


def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 样本数
    # 为所有的分类类目创建字典
    labelCounts ={}
    for featVec in dataSet:
        currentLable=featVec[-1]  # 取得当前数据的标签
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable]=0
        labelCounts[currentLable]+=1 #统计每个类所含样本个数

    #计算香农熵H(x)
    shannonEnt=0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 定义按照某个特征进行划分的函数splitDataSet
# 输入三个变量（待划分的数据集，特征index，分类值）
def splitDataSet(dataSet, feature_index, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[feature_index] == value:
            # reduceFeatVec为去除此特征后的样本信息
            reduceFeatVec = featVec[:feature_index]
            reduceFeatVec.extend(featVec[feature_index+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet #返回不含划分特征的子集


# 定义按照最大信息增益划分数据的函数
def chooseBestFeatureToSplit(dataSet, criterion):
    numFeature = len(dataSet[0]) - 1  #特征数 = 一条样本的长度-1（标签）
    baseEntropy = calcShannonEnt(dataSet)  # 香农熵
    bestInforGain = 0
    bestFeature_index = -1
    for i in range(numFeature):  # 遍历所有特征
        featList = [number[i] for number in dataSet]  # 得到某个特征下所有值（某列）
        uniqualVals = set(featList)  # set无重复的属性特征值
        newEntropy = 0
        splitInfo = 0.0 # 训练集关于特征i的值的熵 

        for value in uniqualVals:
            subDataSet = splitDataSet(dataSet, i, value)  # 按此特征划分数据集
            prob = len(subDataSet) / float(len(dataSet))  # 即p(t)
            newEntropy += prob * calcShannonEnt(subDataSet)  # 对各子集香农熵求和

        if criterion == 'ID3':  # ID3算法使用信息增益作为划分标准 C4.5算法使用信息增益作为划分标准
            infoGain = baseEntropy-newEntropy  # 计算信息增益
        elif criterion == 'C4.5':
            infoGain = (baseEntropy - newEntropy) / splitInfo  #求出第i列属性的信息增益率

        # 求最大信息增益
        if infoGain > bestInforGain:
            bestInforGain = infoGain
            bestFeature_index = i
    return bestFeature_index  # 返回特征值


#投票表决代码
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # print(classCount)
    # print(classCount.items())
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, criterion):
    print('dataSet', dataSet)
    print('label', labels)

    classList = [example[-1] for example in dataSet]

    #类别相同，停止划分
    if classList.count(classList[-1]) == len(classList):
        return classList[-1]

    # print("classList", classList)
    # print("dataSet",dataSet)
    # 长度为1，返回出现次数最多的类别
    if len(dataSet[0]) == 1:
        # print(dataSet)
        # print(classList)
        return majorityCnt(classList)

    #按照信息增益最高选取分类特征属性
    bestFeat_index = chooseBestFeatureToSplit(dataSet, criterion)  # 返回分类的特征序号
    bestFeatLable = labels[bestFeat_index]  # 该特征的label
    myTree = {bestFeatLable: {}}  # 构建树的字典
    print('bestFeat = ', labels[bestFeat_index])
    del(labels[bestFeat_index])  # 从labels的list中删除该label
    featValues = [example[bestFeat_index] for example in dataSet]
    uniqueVals = set(featValues)

    for value in uniqueVals:
        print('value', value)
        subLables = labels[:]  # 子集合
        #构建数据的子集合，并进行递归
        #print("splitDataSet", splitDataSet(dataSet, bestFeat_index, value))
        myTree[bestFeatLable][value] = createTree(splitDataSet(dataSet, bestFeat_index, value), subLables, criterion='ID3')
    return myTree


#输入三个变量（决策树，属性特征标签，测试的数据）
def classify(inputTree, featLables, testVec):
    firstStr = list(inputTree.keys())[0]  # 获取树的第一个特征属性
    secondDict = inputTree[firstStr]  # 树的分支，子集合Dict
    featIndex = featLables.index(firstStr)  # 获取决策树第一层在featLables中的位置
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                return classify(secondDict[key], featLables, testVec)
            else:
                return secondDict[key]


def storeTree(inputTree,filename):
    fw=open(filename, 'wb')  # pickle默认方式是二进制，需要制定'wb'
    pickle.dump(inputTree,fw)
    fw.close()


def grabTree(filename):
    fr=open(filename,'rb')  # 需要制定'rb'，以byte形式读取
    return pickle.load(fr)


def main():
    myDat, labels = createDataSet()
    # print(myDat, feature)
    shanVal = calcShannonEnt(myDat)
    # print(shanVal)
    myTree = createTree(myDat, labels, criterion='C4.5')
    # print("myTree", myTree)
    storeTree(myTree, 'classifierStorage.txt')
    t = grabTree('classifierStorage.txt')
    if t is not None:
        print(t)


if __name__ == '__main__':
    main()