from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 2.0], [1.2, 0.1], [0.1, 1.4], [0.3, 3.5]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify(input,dataSet,label,k):
    dataSize = dataSet.shape[0]

    #欧式距离
    diff = tile(input,(dataSize,1))-dataSet   #每个元素减
    sqdiff=diff**2  #然后平方
    squareDist=sum(sqdiff,axis=1)  #再相加
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

def fileToMatrix(filename):
    fr = open(filename)
    arrOfLines = fr.readlines()
    numberOfLines = len(arrOfLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index=0
    for line in arrOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals-minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet -tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.1   #选用多少数据进行测试，而其他的就是训练集1-hoRatio
    datingDataMat,datingLabels = fileToMatrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    print('numTestVecs = ',numTestVecs)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    print(        errorCount)

def classifyPerson():
    resultList = ['not','small','large']
    percentTats = float(input("percentage of time games"))
    ffMiles = float(input("consume per year"))
    iceCream = float(input("ice per year"))
    datingDataMat,datingLabel = fileToMatrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)

    #期望的，假设输入是[10,10000,0.4]
    inArr = array([ffMiles,percentTats,iceCream])

    #对输入的期望值进行归一,(inArr-minVals)/ranges

    #根据输入的期望来对测试数据进行分类
    classifierResult = classify((inArr-minVals)/ranges,normMat,datingLabel,3)
    print("result:%s"%resultList[classifierResult-1])

# usage:
# import sys
# sys.path.append("...文件路径...")
# import knn
# from numpy import *
# dataSet,labels = knn.createDataSet()
# input = array([1.1,0.3])
# K = 3
# output = knn.classify(input,dataSet,labels,K)
# print("测试数据为:",input,"分类结果为：",output)