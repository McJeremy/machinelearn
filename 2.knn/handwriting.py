#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/27 0027 16:45
# @Author  : xuzz
# @File    : handwriting.py
# @Software: PyCharm
from os import listdir

from numpy import *
import operator

def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0]

    #欧式距离
    # x-y
    diff = tile(input,(dataSize,1))-dataSet   #每个元素减
    # (x-y)^2
    sqdiff=diff**2  #然后平方
    # (x-y)^2+(x-y)^2.....
    squareDist=sum(sqdiff,axis=1)  #再相加
    # ((x-y)^2+(x-y)^2.....)^0.5
    dist=squareDist**0.5   #开平方

    ##对距离进行排序
    sortedDistIndex = argsort(dist)  ##argsort()根据元素的值从大到小对元素进行排序，返回下标

    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndex[i]]
        ###对选取的K个样本所属的类别个数进行统计
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    ###选取出现的类别次数最多的类别
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            classes = key

    return classes

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):  #循环读出文件的前32行
        lineStr = fr.readline()
        for j in range(32):  #每行的头32个字符
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'%fileNameStr)
        classifierResult = classify(vectorUnderTest,trainingMat,hwLabels,3)
        print('测试返回结果:%d,真正的结果是:%d'%(classifierResult,classNumStr))
        if classifierResult!=classNumStr: errorCount+=1.0

    print("total error:%d"%errorCount)
    print("error rate is :%f"%(errorCount/float(mTest)))


