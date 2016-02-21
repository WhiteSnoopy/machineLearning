#-*-coding:utf-8-*-
import sys
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
reload(sys)
sys.setdefaultencoding('utf-8')
import operator
from os import listdir
from kNN import classify0
#手写识别系统
#listdir获得指定目录中的内容的文件名
#2.3.1准备数据：将图像转换为测试向量 将32x32的二进制图像矩阵转换为1x1024的向量
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect
#2.3.2测试算法: 使用K-近邻算法识别手写数字
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorcount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print 'the classifier came back with:%d,the real answer is :%d' % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorcount += 1.0

    print "\nThe local number of errors is :%d" % errorcount
    print "\nThe total error rate is: %f " % (errorcount/float(mTest))


def main():
    #returnVect = img2vector('0_0.txt')
    #print returnVect[0, 0:31]
    handwritingClassTest()

if __name__=='__main__':
    main()
