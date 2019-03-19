import csv
import random
import math
import numpy as np
from sklearn.naive_bayes import GaussianNB


def loadCsv(filename):
    lines = csv.reader(open(filename))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset)*splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def separateByClass(dataset):
    separated = {}
    count = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if vector[-1] not in separated:
            separated[vector[-1]] = []
            count[vector[-1]] = 0
        separated[vector[-1]].append(vector)
        count[vector[-1]] += 1
    return separated, count


def mean(numbers):
    return sum(numbers)/float(len(numbers))


def stdev(numbers):
    avg = mean(numbers)
    variance = sum(pow(x-avg, 2) for x in numbers) / float(len(numbers)-1)
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separaed, count = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separaed.items():
        summaries[classValue] = summarize(instances)
    return summaries, count


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector, count):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = count[classValue] / 119
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector, count):
    probabilities = calculateClassProbabilities(summaries, inputVector, count)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestLabel = classValue
            bestProb = probability
    return bestLabel


def getPredictions(summaries, testSet, count):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i], count)
        predictions.append(result)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


if __name__ == '__main__':
    filename = 'wine.data'
    splitRatio = 0.67
    # 加载数据
    dataset = loadCsv(filename)
    # 将加载进来的list类型数据转为numpy类型数据，便于处理
    dataset_array = np.array(dataset)
    # 示例实验中最后一维是标签数据，wine.data中第0维是标签数据，将标签数据换到最后一维
    dataset_array[:, [0, -1]] = dataset_array[:, [-1, 0]]
    # 再将处理好的numpy数据转为list类型数据
    dataset = dataset_array.tolist()
    # 按splitRatio的比例划分数据集，67%的数据作为训练集，33%的数据作为测试集
    trainingSet, testSet = splitDataset(dataset, splitRatio)

    # 将训练集的数据按类别划分为summaries
    # summaries是一个dirc类型数据，key表示类别，value是一个list，每一个元素表示属于这个类别的一个样本
    summaries, count = summarizeByClass(trainingSet)
    # 按类别生成高斯模型，并且将testSet的数据放入，将概率最大的记为预测类别
    predictions = getPredictions(summaries, testSet, count)
    # 将预测类别与真实类别进行比对，计算准确率
    accuracy = getAccuracy(testSet, predictions)
    print('accuracy: {}%'.format(accuracy))


    print(count)

    # 使用GaussianNB
    # 将训练集和测试集转为numpy数组
    trainingSet_array = np.array(trainingSet)
    testSet_array = np.array(testSet)

    # X_train为训练集输入部分，y_train为训练集的标签部分
    X_train = trainingSet_array[:, :-1]
    y_train = trainingSet_array[:, -1]

    # X_test为测试集输入部分，y_test为测试集的标签部分
    X_test = testSet_array[:, :-1]
    y_test = testSet_array[:, -1]

    clf = GaussianNB().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    correct = 0
    for i in range(len(y_test)):
        if y_pred[i] == y_test[i]:
            correct += 1
    accuracy_sk = correct / len(y_test) * 100.0
    print('accuracy_sk-learn: {}%'.format(accuracy_sk))
