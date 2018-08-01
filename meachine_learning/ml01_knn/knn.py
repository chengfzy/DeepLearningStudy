import numpy as np
import operator
from common.debug_info import *


def classify(x, dataset, labels, k):
    """
    k neighbor nearest algorithm
    :param x:       input data value, vector to compare to existing dataset (1xN)
    :param dataset: size m data set of known vector(N x M)
    :param labels:  data set labels (1xM vector)
    :param k:       number of neighbors use for comparison(should be an odd number)
    :return:        the most popular class label
    """
    diff_mat = x - dataset
    square_diff_mat = diff_mat ** 2
    square_dist = square_diff_mat.sum(axis=1)
    distance = square_dist ** 0.5
    sorted_dist_idx = distance.argsort()
    class_count = {}
    for i in range(k):
        vote_label = labels[sorted_dist_idx[i]]
        class_count[vote_label] = class_count.get(vote_label, 0) + 1
    sorted_classs_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_classs_count[0][0]


class SimpleKnnClassify:
    """
    Simple KNN Test, for input several data
    """

    def test(self):
        group, labels = self.__create_data_set()
        print('goups = ', group)
        print('labels = ', labels)
        print('knn result = ', classify(np.array([0, 0]), group, labels, 3))

    # create data sets
    @staticmethod
    def __create_data_set():
        group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels = ['A', 'A', 'B', 'B']
        return group, labels


class DatingMatch:
    """
    Improving matches from a dating site with kNN
    """

    __filename = ""
    __test_ratio = 0  # test data ratio in whole data set

    def __init__(self, filename, test_ratio=0.10):
        self.__filename = filename
        self.__test_ratio = test_ratio

    def test(self):
        """
        load the data set from file and train, and then take few data for test
        :return:
        """
        data, labels = self.__load_data()
        norm_data, ranges, min_vals = self.__auto_norm(data)
        m = norm_data.shape[0]
        test_num = int(m * self.__test_ratio)
        error_count = 0.0
        for i in range(test_num):
            result = classify(norm_data[i, :], norm_data[test_num: m, :], labels[test_num: m], 3)
            print("(classifier result, real answer) = ({0}, {1})".format(result, labels[i]), end='')
            if result != labels[i]:
                print("\tERROR")
                error_count += 1
            else:
                print("")

        print("the total error rate is: {0}".format(error_count / test_num))
        print("error count = ", error_count)

    def __load_data(self):
        """
            load data from file and return data and label, each line include 4 data:
            1. number of frequent flyer miles earned per year
            2. percentage of time spent playing video games
            3. liters of ice cream consumed per week
            4. labels: she didn't like, liked in small doses, liked in large doses
            :return: data and lables
        """
        file = open(self.__filename)
        array_lines = file.readlines()
        number_of_lines = len(array_lines)
        data = np.zeros((number_of_lines, 3))  # data
        labels = []  # labels
        index = 0
        for line in array_lines:
            line = line.strip()
            list_from_line = line.split('\t')
            data[index, :] = list_from_line[0:3]
            labels.append(int(list_from_line[-1]))
            index += 1
        return data, labels

    @staticmethod
    def __auto_norm(dataset):
        """
        normalize the data set
        :param dataset: input data set
        :return:  normalized data set, range and min values
        """
        min_vals = dataset.min(0)
        max_vals = dataset.max(0)
        ranges = max_vals - min_vals
        m = dataset.shape[0]
        norm_dataset = dataset - np.tile(min_vals, (m, 1))
        norm_dataset = norm_dataset / np.tile(ranges, (m, 1))  # element wise divide
        return norm_dataset, ranges, min_vals


if __name__ == '__main__':
    # Simple KNN Test
    print(section("Simple KNN Test"))
    simple_classify = SimpleKnnClassify()
    simple_classify.test()

    # Dating Matching
    print(section("Dating Match"))
    dating_match = DatingMatch('../../data/dating/datingTestSet2.txt', 0.1)
    dating_match.test()
