# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 22:46:00 2019

@author: lianxiang
"""

import numpy as np


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


def cost_function(theta, x, y, m):
    '''Cost function J definition.'''
    diff = np.dot(x, theta) - y
    return (1./2*m) * np.dot(diff.T, diff)


def optimizer_function(theta, x, y):
    '''Optimizert of the function J definition.'''
    diff = np.dot(x, theta) - y
    return (1./m) * np.dot(x.T, diff)


def analytical_method(x, y, theta):
    '''Analytical method.'''
    theta = np.dot(np.dot(np.linalg.inv(np.dot(x.T, x)), x.T), y)
    # theta = ((x.T * x).-1) * x.T * y
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
frequency=1000


theta = analytical_method(x, y, theta)