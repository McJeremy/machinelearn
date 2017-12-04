#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/28 0028 13:12
# @Author  : xuzz
# @File    : predict_house_price.py
# @Software: PyCharm
import numpy as np
import pandas as pd

def loadDataset():
    data = pd.read_csv('input.csv')
    x_param = []
    y_param = []
    for squre,price in zip(data['square_feet'],data['price']):
        x_param.append(squre)
        y_param.append(price)
    return np.matrix([x_param,y_param])

def cost_function(x,y,theta):
    '''
    成本函数
    :param x: 测试
    :param y: 实际
    :param theta:
    :return:
    '''
    h = np.mat(x*theta)
    m = x.shape[1]
    s = 0
    return (1/(2*m))*((h-y)*((h-y).T))
