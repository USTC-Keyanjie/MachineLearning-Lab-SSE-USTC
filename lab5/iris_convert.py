# -*- coding: utf-8 -*-
# @File     : iris_convert.py
# @Author   : Yanjie Ke
# @Date     : 2019/1/13 12:50
# @License  : Copyright(C), USTC

import random
from sklearn import datasets


def main():
    iris = datasets.load_iris()
    iris_feature, iris_target = iris.data, iris.target
    print(iris_target)

    iris_train = open('iris_data_train.txt', 'w')
    iris_test = open('iris_data_test.txt', 'w')

    for i in range(len(iris_feature)):
        target = iris_target[i]
        feature = ''
        for j in range(len(iris_feature[i])):
            feature += str(j+1) + ':' + str(iris_feature[i][j]) + ' '
        line = str(target) + ' ' + feature[:-1] + '\n'
        if random.randint(0, 10) < 3:
            iris_test.write(line)
        else:
            iris_train.write(line)

    iris_train.close()
    iris_test.close()


if __name__ == '__main__':
    main()