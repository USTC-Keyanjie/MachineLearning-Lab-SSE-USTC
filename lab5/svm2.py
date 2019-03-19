# -*- coding: utf-8 -*-
# @File     : svm2.py
# @Author   : Yanjie Ke
# @Date     : 2019/1/13 12:36
# @License  : Copyright(C), USTC

from svmutil import *

print('ex8a.txt')
y, x = svm_read_problem('libsvm-3.23/python/ex8Data/ex8a.txt')

split_num = int(len(y) * 0.66)
train = svm_problem(y[:split_num], x[:split_num], isKernel=True)

'''
gamma = 1 / (2 * σ^2)
'''
gamma = 650
model = svm_train(train, '-s 0 -t 2 -g %s' % gamma)
svm_predict(y[split_num:], x[split_num:], model)

print('ex8b.txt')
y, x = svm_read_problem('libsvm-3.23/python/ex8Data/ex8b.txt')

split_num = int(len(y) * 0.66)
train = svm_problem(y[:split_num], x[:split_num], isKernel=True)

'''
gamma = 1 / (2 * σ^2)
'''
gamma = 650
model = svm_train(train, '-s 0 -t 2 -g %s' % gamma)

svm_predict(y[split_num:], x[split_num:], model)


