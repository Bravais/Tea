# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:05:08 2019

@author: lianxiang
"""

import numpy as np
import pandas as pd

x = np.arange(1000).reshape(100,10)
y = np.random.randint(1,5,size=(100, 1))

class_num = 4
'''
def yMultiClass(x, y, class_num=1):
    #y multi-class classfication
    Y = []
    for i in range(1,class_num+1):
        temp_y = np.where(y==i, 1, 0)
        Y.append(temp_y)
    return Y
'''
def yMultiClass(x, y, class_num=1):
    '''y multi-class classfication and the value of temp_y is matrix style'''
    c = np.ones(class_num)
    temp_y = y * c
    temp_y = np.where(temp_y == np.arange(1,class_num+1), 1, 0)
    return temp_y