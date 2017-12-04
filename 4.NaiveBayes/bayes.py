#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/11/1 0001 12:52
# @Author  : xuzz
# @File    : bayes.py
# @Software: PyCharm

#http://blog.csdn.net/moxigandashu/article/details/71480251?locationNum=16&fps=1

from  numpy import *

def loadDataSet():
    """
    创建数据集
    :return: 单词列表postingList, 所属类别classVec
    """
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'gar e'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

def createVocabList(dataSet):
    """
    获取所有单词的集合
    :param dataSet: 数据集
    :return: 所有单词的集合(即不含重复元素的单词列表)
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        # 操作符 | 用于求两个集合的并集
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    """
    遍历查看该单词是否出现，出现该单词则将该单词置1
    根据传入的inputSet，查看里面的单词是否在vocabList中出现，如果出现了，标记为1，否则标记为0
    :param vocabList: 所有单词集合列表
    :param inputSet: 输入数据集
    :return: 匹配列表[0,1,0,1...]，其中 1与0 表示词汇表中的单词是否出现在输入的数据集中
    """
    # 创建一个和词汇表等长的向量，并将其元素都设置为0
    # 词汇表就是vocabList
    returnVec = [0] * len(vocabList)# [0,0......]

    # 遍历文档中的所有单词，如果出现了词汇表中的单词，则将输出的文档向量中的对应值设为1
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # [0,0......]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    '''
  计算条件概率
    :param trainMatrix:
    :param trainCategory:
    :return:
    '''
    '''
       In [111]: reload(bayes)
     ...:
Out[111]: <module 'bayes' from 'G:\\Workspaces\\MachineLearning\\bayes.py'>
In [112]: listPosts,listClasses=bayes.loadDataSet()
     ...:
In [113]: myVocabList=bayes.createVocabList(listPosts)
     ...:
In [114]: trainMat=[]
     ...:
In [115]: for postinDoc in listPosts:
     ...:     trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
     ...:
     ...:
In [116]: p0v,p1v,pAb=bayes.trainNB0(trainMat,listClasses)
     ...:
In [117]: pAb
     ...:
Out[117]: 0.5
In [118]: p0v
     ...:
Out[118]:
array([ 0.04166667,  0.        ,  0.08333333, ...,  0.04166667,
        0.04166667,  0.        ])
In [119]: p1v
     ...:
Out[119]:
array([ 0.        ,  0.05263158,  0.05263158, ...,  0.        ,
        0.05263158,  0.05263158])
#myVocabList中第26个词汇是'love'，即myVocabList[25]='love'
In [121]: p0v[25]
Out[121]: 0.041666666666666664
In [122]: p1v[25]
Out[122]: 0.0
##myVocabList中第13个词汇是'stupid'，即myVocabList[13]='stupid'
In [124]: p0v[12]
Out[124]: 0.0
In [125]: p1v[12]
Out[125]: 0.15789473684210525
从结果我们看到，侮辱性文档出现的概率是0.5，词项’love’在侮辱性文档中出现的概率是0，在正常言论中出现的概率是0.042；词项‘stupid’在正常言论中出现的概率是0，在侮辱性言论中出现的规律是0.158.
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]  #注意p1Num相加后，还是向量
            p1Denom+=sum(trainMatrix[i])   #相加后，是数值
        else :
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect = log(p1Num /p1Denom)
    p0Vect = log(p0Num/p0Denom)

    '''
    1.pAbusive=sum(trainCategory)/len(trainCategory)，表示文档集中分类为1的文档数目，累加求和将词向量中所有1相加，len求长度函数则对所有0和1进行计数，最后得到分类为1的概率
2.p0Num+=trainMatrix[i];p0Demon+=sum(trainMatrix[i])，前者是向量相加，其结果还是向量，trainMatrix[i]中是1的位置全部加到p0Num中，后者是先求和（该词向量中词项的数目），其结果是数值，表示词项总数。
3.p0Vec=p0Num/p0Demon，向量除以数值，结果是向量，向量中每个元素都除以该数值。'''
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    listPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listPosts)
    trainMat = []
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = setOfWords2Vec(myVocabList, testEntry)
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry, 'classified as:', classifyNB(thisDoc, p0V, p1V, pAb))

