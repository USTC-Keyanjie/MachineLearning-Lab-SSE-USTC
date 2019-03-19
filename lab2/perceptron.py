# -*- coding: utf-8 -*-
import copy
import pandas as pd
import numpy as np


names = ("age, workclass, fnlwgt, education, education-num, "
         "marital-status, occupation, relationship, race, sex, "
         "capital-gain, capital-loss, hours-per-week, "
         "native-country, income").split(', ')
data = pd.read_csv('adult.data', names=names)

col_names = data.columns.tolist()
target = data['income']
features_data = data.drop('income', axis=1)
# 提取数值类型为整数或浮点数的变量
numeric_features = [c for c in features_data if features_data[c].dtype.kind in ('i', 'f')]
print(numeric_features)
numeric_data = features_data[numeric_features]
print(type(features_data))
categorical_data = features_data.drop(numeric_features, 1)
categorical_data.head(5)

# pd.factorize即可将分类变量转换为数值表示
# apply运算将转换函数应用到每一个变量维度
categorical_data_encoded = categorical_data.apply(lambda x: pd.factorize(x)[0])
categorical_data_encoded.head(5)
features = pd.concat([numeric_data, categorical_data_encoded], axis=1)
print(type(features))
features = features.drop(["education", "capital-loss", "relationship"], 1)
features.head()

# 转换数据类型
X = features.values.astype(np.float32)
y = np.zeros(X.shape[0])
# 收入水平 ">50K" 记为1，“<=50K” 记为0
for i in range(y.shape[0]):
    if target[i] == ' >50K':
        y[i] = 1
    else:
        y[i] = -1


w = [0] * X.shape[1]
b = 0
eta = 0.5
record = []


# sign是符号函数 res>=0时为1，否则为-1
def sign(X, y):
    global w, b
    sum_wx = 0
    for i in range(len(X)):
        sum_wx += w[i] * X[i]
    res = y * (sum_wx + b)
    if res > 0:
        return 1
    else:
        return -1


def update(X, y):
    global w, b, record
    for i in range(len(X)):
        w[i] = w[i] + eta * y * X[i]

    b = b + eta * y
    record.append([copy.copy(w), b])


def perceptron():
    count = 1
    for i in range(X.shape[0]):
        flag = sign(X[i], y[i])
        if not flag > 0:  # 错分类
            # count = 1
            update(X[i], y[i])
        else:
            count += 1

    print("Accurary:{:.2f}%".format(count / X.shape[0] * 100))
    if count >= len(data):
        return 1
    else:
        return -1


# 标准化数据集 X
def standardize(X):
    X_std = np.zeros(X.shape)
    # 每一列求均值
    mean = X.mean(axis=0)
    # 每一列求标准差
    std = X.std(axis=0)
    # print(mean)
    # print(std)

    for col in range(np.shape(X)[1]):
        # 感觉这里写错了，应该是：
        X_std[:, col] = (X[:, col] - mean[col]) / std[col]
        # X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]

    return X_std


if __name__ == '__main__':
    X = standardize(X)
    while 1:
        if perceptron() > 0:
            break
    print(record)
