# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:46:00 2019

@author: lianxiang
"""

import numpy as np
import matplotlib.pyplot as plt


def cost_function(theta, x, y, m):
    '''Cost function J definition.'''
    z = np.dot(x, theta)
    hz = 1. / (1+np.exp(-1*z))
    return (-1./m) * (y*np.log10(hz)+(1-y)*np.log10(1-hz)).sum


def optimizer_function(theta, x, y):
    '''Optimizert of the function J definition.'''
    z = np.dot(x, theta)
    hz = 1. / (1+np.exp(-1*z))
    diff = hz - y
    return (1./m) * np.dot(x.T, diff)


def gradient_descent(x, y, alpha, theta, frequency=1000):
    '''Gradient descent algorithm.'''
    j = []
    for i in range(frequency):
        diff = optimizer_function(theta, x, y)
        theta = theta - alpha * diff
        
        '''Figure the change of J(theta)'''
        temp = cost_function(theta, x, y, m)
        if temp < 500: # 500可以进行调整
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

theta = gradient_descent(x, y, alpha, theta, frequency)