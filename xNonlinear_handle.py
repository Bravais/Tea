# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 17:56:53 2019

@author: lianxiang
"""

import numpy as np 
import pandas as pd

x0 = np.ones((8,1))
x1 = np.random.randn(24).reshape((8,3))
x = np.hstack((x0, x1))

def nonLineTranslate(x, layer=1):
    '''Data nonlinear processing'''
    a1, b1 = x.shape
    x_1 = x.reshape((a1,1,b1))
    for i in range(layer-1):
        a2, b2 = x.shape
        x_2 = x.reshape((a2,b2,1))
        x_temp = x_1 * x_2
        x_temp = x_temp.flatten().reshape((a1,b1*b2))
        x_temp = pd.DataFrame(x_temp.T).drop_duplicates()
        x = x_temp.values.T
    return x

x1 = nonLineTranslate(x, 2)