# -*- coding: utf-8 -*-
# @File     : svm1.py
# @Author   : Yanjie Ke
# @Date     : 2019/1/13 12:36
# @License  : Copyright(C), USTC

from svmutil import *

y, x = svm_read_problem('libsvm-3.23/python/ex7Data/email_train-all.txt')
y_test, x_test = svm_read_problem('libsvm-3.23/python/ex7Data/email_test.txt')

'''
-s svm类型：SVM设置类型(默认0)
    0 -- C-SVC
    1 -- nu-SVC
    2 -- one-class SVM
    3 -- epsilon-SVR
    4 -- nu-SVR

-t 核函数类型：核函数设置类型(默认2)
    0 –- 线性：u'v
    1 –- 多项式：(r*u'v + coef0)^degree
    2 –- RBF函数：exp(-r|u-v|^2)
    3 –- sigmoid：tanh(r*u'v + coef0)

-g r(gamma)：核函数中的gamma函数设置(针对多项式/rbf/sigmoid核函数)

-c cost：设置C-SVC，e -SVR和v-SVR的参数(损失函数)(默认1)，惩罚系数

-n nu：设置v-SVC，一类SVM和v- SVR的参数(默认0.5)

-p p：设置e -SVR 中损失函数p的值(默认0.1)

-d degree：核函数中的degree设置(针对多项式核函数)(默认3)

-wi weight：设置第几类的参数C为weight*C(C-SVC中的C)(默认1)

-v n: n-fold交互检验模式，n为fold的个数，必须大于等于2
'''
model = svm_train(y, x, '-s 0 -t 0 -c 1')

svm_predict(y_test, x_test, model)
