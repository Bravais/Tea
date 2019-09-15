# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 14:34:29 2019

@author: lianxiang
"""

import numpy as np
import pandas as pd


m = 20
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
x = np.hstack((X0, X1))

def rescaling(x):
    '''Rescaling [0, 1]'''
    x = pd.DataFrame(x)
    x_min = x.min()
    x_max = x.max()
    return ((x-x_min) / (x_max-x_min)).values


def rescaling2(x):
    '''Rescaling [-1, 1]'''
    x = pd.DataFrame(x)
    x_mean = x.mean()
    x_max = x.max()
    return ((x-x_mean) / (x_max-x_mean)).values


def standardization(x):
    '''standardization zero-mean and unit-variance in SVM, Logistic regression and Neural Networks'''
    x = pd.DataFrame(x)
    x_mean = x.mean()
    x_std = x.std()
    return ((x-x_mean) / x_std).values


def mean_normalization(x):
    '''mean_normalization [-0.5, 0.5]'''
    x = pd.DataFrame(x)
    x_mean = x.mean()
    x_max = x.max()
    x_min = x.min()
    return ((x-x_mean) / (x_max-x_min))