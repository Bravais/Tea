# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:46:00 2019

@author: lianxiang
"""

import numpy as np
import matplotlib.pyplot as plt

'''
def gradientDescent(x, sita, alpha, y, m):
    for i in range(7000):
        temp_sita = sita.copy()
        differential = np.dot(np.dot(temp_sita, x.T)-y, x) / m
        sita = sita - alpha * differential
        if all(abs(differential) <= 1e-5):
            break
        print(sita)
    # return sita

'''
def cost_function(theta, liambda, x, y, m):
    '''Cost function J definition.'''
    diff = np.dot(x, theta) - y
    return (1./2*m) * (np.dot(diff.T, diff) + liambda*np.dot(theta.T, theta))[0][0]


def optimizer_function(theta, x, y):
    '''Optimizert of the function J definition.'''
    diff = np.dot(x, theta) - y
    return (1./m) * np.dot(x.T, diff)


def gradient_descent(x, y, liambda, alpha, theta, frequency=1000):
    '''Reqularization gradient descent algorithm.'''
    j = []
    for i in range(frequency):
        diff = optimizer_function(theta, x, y)
        theta = np.vstack((theta[[0]], theta[1:]*(1-alpha*liambda/m)))
        theta = theta - alpha * diff
        
        '''Figure the change of J(theta)'''
        temp = cost_function(theta, liambda, x, y, m)
        if temp < 1000: # 500可以进行调整
            j.append(temp)
        
        if all(abs(diff) < 1e-5):
            break
    # print(theta)
    plt.plot(j)
    plt.show()
    return theta


m = 20
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
x = np.hstack((X0, X1))
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape((m,1))
theta = np.array([10,10]).reshape((2,1))
alpha = 0.01
frequency=3000
liambda = 31 # 惩罚参数
theta = gradient_descent(x, y, liambda, alpha, theta, frequency)