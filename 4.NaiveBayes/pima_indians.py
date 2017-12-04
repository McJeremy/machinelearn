#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/17 0017 8:49
# @Author  : xuzz
# @File    : pima_indians.py
# @Software: PyCharm
import pandas as pd
import random

#http://python.jobbole.com/81019/

def loadCsv(fileName):
    lines =  pd.read_csv(fileName)
    # print(lines)
    dataset = lines
    # for i in range(len(dataset)):
    #     dataset[i] = [float(x) for x in dataset[i]]
    return dataset

def splitDataset(dataSet,splitRatio):
    trainSize = int(len(dataSet)*splitRatio)
    trainSet=[]
    copy = list(dataSet)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]