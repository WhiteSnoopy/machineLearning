#-*-coding:utf-8-*-
from numpy import array
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
import operator
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def createDataSet():
    #创建二维数组
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#inX 用于分类的输入向量
#输入的训练样本集dataSet
#labels标签向量
#k用于选择最近邻居的数目
#其中标签向量的元素数目和矩阵的行数相同
#shape函数是numpy.core.fromnumeric中的函数，它的功能是读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度
def classify0(inX, dataSet, labels, k):
    dataSetsize = dataSet.shape[0]
    #tile(A,reps)
    diffMat = tile(inX, (dataSetsize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistIndicies = distance.argsort()
    classCount ={}
    #选择距离最小的K个点
    #确定前K个最小元素所在的主要分类
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

##################################################################################################
#使用kNN改进约会网站的配对效果
'''
1.收集数据(收集数据)
2.准备数据(使用python解析文本文件) 文本文件存放在datingTestSet.txt
'''
#2.2.1从文本文件中解析数据
#将待处理数据的格式转变为分类器可以接受的格式
def file2matrix(filename):
    fr = open(filename)
    arrayOline = fr.readlines()
    numberOfLines = len(arrayOline)
    #创建二维数组
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOline:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


#2.2.2分析数据,使用matplotlib创建散点图
def scatterDiagram(datingDataMat, datingLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 150.*array(datingLabels), 15.0*array(datingLabels))
    plt.show()

#2.2.3准备数据:归一化数值(用于处理不同取值范围的数值，将取值范围处理为0-1或者是-1-1) neValue = (oldValue-min)/(max-min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxvals = dataSet.max(0)
    ranges = maxvals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

#2.2.4测试算法：作为完整程序验证分类器 90%作为训练样本来训练分类器，10%测试分类器
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels=file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
        print "the classifier came back with :%d, the real answer is :%d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is :%f" % (errorCount/numTestVecs)


#2.2.5使用算法：构建完整可用的系统
def classifyPerson():
    resultList = ['not at all', 'in small does', 'in large does']
    percentTats = float(raw_input("percentage of time spent playing video games?"))
    ffMiles = float(raw_input("frequent flier miles earned per year"))
    iceCream = float(raw_input("liters of iceCream consumed per year"))
    datingDataMat, datingLabels=file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print "you will probably like this person:", resultList[classifierResult-1]

def main():
    datingDataMat, datingLabels=file2matrix('datingTestSet2.txt')
    #scatterDiagram(datingDataMat, datingLabels)
    #datingClassTest()
    classifyPerson()


if __name__=='__main__':
    #group, labels = createDataSet()
    #print classify0([0,0], group, labels, 30)
    main()