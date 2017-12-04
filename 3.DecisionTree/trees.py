#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/10/30 0030 17:35
# @Author  : xuzz
# @File    : trees.py.py
# @Software: PyCharm
import operator
from math import log

# 计算香农熵'
def calcShannonEnt(dataSet):
    '''
            as:
        nosurfacing  fliper  class
     1    1             1       yes
     2    1             1       yes
     3    1              0       n0
     4    0              1       no
     5    0              1       no

    '''
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)

    return shannonEnt


def createDataSet():
    # 判断是否鱼类
    # no surfacing -> 不浮出水面是否可以生存
    # flipper ->是否有脚蹼
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flipper']

    '''
        as:
        nosurfacing  fliper  class
     1    1             1       yes
     2    1             1       yes
     3    1              0       n0
     4    0              1       no

    '''
    return dataSet,labels

#1 = my1
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVect in dataSet:
        if featVect[axis]==value:
            #从0开始取axis个元素，刚好不含axis自己
            reducedFectVect = featVect[:axis]
            #从axis下一个元素开始取完并合并
            reducedFectVect.extend(featVect[axis+1:])
            retDataSet.append(reducedFectVect)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):

    #所有特征长度
    numFeatures = len(dataSet[0])-1

    #基础熵，样本的熵
    baseEntropy = calcShannonEnt(dataSet)

    bestInfoGain = 0.0
    bestFeature = -1

    ''' eg:
    假设有样本：
    特征1:  A    A   B   A   C  C   C  B
    特征2：
    结论：  是   否   是  是  否 是  是  否

    可以计算出样本的总熵：
       1、结论只有两种：是或否，总共有8个，其中是有5，否有6
       2、那么总熵= -( 5/8 * log(5/8,2)  + 5/8 * log(5/8,2) )
                =0.847589.............

    接下来循环处理每一个特征，以特征1为例进行：
    1、得到特征1的值：[A A B A C C C B]
    2、去重后为：[A B C ]
    3、循环处理每一个特征值（节点）：
      3.1、得到节点的权重：A = 3/8  B = 2/8 C = 3/8
           因为总共有8个节点，A有3个....
      3.2、对每个节点计算熵
         3.2.1、先拆分特征值得到每个节点的数据集，结果为：
             A->[是，否，是]  B->[是，否] C->[否，是，是]
         3.2.1、对每一个拆后的数据集计算熵，结果为：
             h(A)= -(2/3 *log(2/3,2) + 1/3 *log(1/3,2)) =0.9183
             h(B)= -(1/2 *log(1/2,2) + 1/2 *log(1/2,2)) =1.0
             h(C)= -(2/3 *log(2/3,2) + 1/3 *log(1/3,2)) =0.9183
      3.3、对权重和熵相加，得到特征1的熵：
         3/8 * 0.91 + 2/8*1 +3/8*0.91 = 0.93
     4、用总熵减去特征的熵，就得到信息增益：0.8475-0.93
    '''

    #循环每个特征，基于信息增益寻找最好的特征
    # 循环“付出水面”、“是否有脚蹼",
    for i in range(numFeatures):
        #得到当前特征的特征值
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # 按特征进行拆分
            '''比如，按是否有脚蹼划分，则得到新的数据样本：
            eg:
          nosurfacing    class
     1    1              yes
     2    1              yes
     3    0              no
            '''
            subDataSet = splitDataSet(dataSet,i,value)

            #并计算拆分后的数据集的总熵
            #  首先计算这个特征下节点的加权值
            prob = len(subDataSet)/float(len(dataSet))

            #  再计算各个特征的熵,加上权值，得到节点的总熵
            newEntropy += prob * calcShannonEnt(subDataSet)

        infoGain = baseEntropy-newEntropy
        if infoGain>bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote]=0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reversed=True)
    return sortedClassCount[0][0]

'''
creatBranch():
    if so return 类标签:
    else:
        寻找划分数据集的最好特征
        划分数据集
        创建分支节点
            for 每个划分的子集
                调用函数 creatBranch() 并增加返回结果到分支节点中
        return 分支节点
'''
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        #类别完全相同，则停止划分
        return classList[0]
    if len(dataSet[0])==1:
        #遍历完所有特征时，返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]

    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree

#input:{'no surfacing': {0: 'no', 1: {'fliper': {0: 'no', 1: 'yes'}}}}
#labels = ['no surfacing','fliper']
#testVec = [1,1]
def classify(inputTree,labels,testVec):
    firstStr = list(inputTree.keys())[0]  # no surfacing
    secondDict = inputTree[firstStr]   # {0: 'no', 1: {'fliper': {0: 'no', 1: 'yes'}}}
    featIndex = labels.index(firstStr)   #0
    #[0,1]
    for key in secondDict.keys():
        if testVec[featIndex] == key :
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict,labels,testVec)
            else: classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree,fileName):
    import pickle
    fw = open(fileName,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(fileName):
    import pickle
    fr = open(fileName)
    return pickle.load(fr)

def lenseTest():
    #收集数据
    fr = open('lenses.txt')
    #准备数据
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    preLensesLabels,lensesLabels = ['age','prescript','astigmatic','tearRate'],['age','prescript','astigmatic','tearRate']
    #分析数据 (使用matplotlib图形分析)
    lensesTree = createTree(lenses,lensesLabels)
    return classify(lensesTree,preLensesLabels,['pre','hyper','no','reduced'])


