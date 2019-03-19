# -*- coding: utf-8 -*-
# @File     : svm_iris.py
# @Author   : Yanjie Ke
# @Date     : 2019/1/13 12:45
# @License  : Copyright(C), USTC

from svmutil import *

y, x = svm_read_problem('iris_data_train.txt')
y_test, x_test = svm_read_problem('iris_data_test.txt')
gamma = 100

model = svm_train(y, x, '-s 0 -t 2 -g %s' % gamma)

svm_predict(y_test, x_test, model)
