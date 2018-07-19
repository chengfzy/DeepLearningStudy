import numpy as np
import operator


# create data sets
def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    """
    k neighbor nearst algorithm
    :param inX:     vector to compare to existing dataset (1xN)
    :param dataSet: size m data set of known vector(N x M)
    :param labels:  data set labels (1xM vector)
    :param k:       number of neighbors use for comparison(should be an odd number)
    :return:        the most popular class label
    """
    dataSetSize = dataSet.shape[0]
    diffMat = inX - dataSet
    sqDiffMat = diffMat ** 2
    sqDist = sqDiffMat.sum(axis=1)
    distance = sqDist ** 0.5
    sortedDistIdx = distance.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIdx[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClasssCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClasssCount[0][0]



if __name__== '__main__':
    group, labels = createDataSet()
    print('goups = ', group)
    print('labels = ', labels)
    print('knn result = ', classify0([0, 0], group, labels, 3))


