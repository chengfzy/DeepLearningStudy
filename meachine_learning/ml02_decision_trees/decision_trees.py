"""
Decision Tree
"""

from math import log


def calc_shannon_ent(dataset):
    """
    Calculate the Shannon entropy of a dataset
    :param dataset: dataset
    :return: Shannon entropy
    """
    num_entries = len(dataset)
    label_counts = {}
    for feature_vec in dataset:
        current_label = feature_vec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries
        shannon_ent -= prob * log(prob, 2)
    return shannon_ent


class SimpleClassify:
    """
    Simple Test
    """

    def test(self):
        """
        Test
        """
        self.__create_dataset()
        print(self.__dataset)
        print(self.__labels)
        print('Shannon entropy: ', calc_shannon_ent(self.__dataset))

        self.__choose_best_feature_to_split()

        # choose best method to split dataset
        print('best feature: ', self.__choose_best_feature_to_split())

    def __create_dataset(self):
        """
        create dataset. labels is the quest for first 2 columns in dataset, the last column in dataset is the answer
        that whether this features is a fish
        """
        self.__dataset = [[1, 1, "yes"],
                          [1, 1, "yes"],
                          [1, 0, "no"],
                          [0, 1, "no"],
                          [0, 1, "no"]]
        self.__labels = ["no surfacing", "flippers"]

    def __split_dataset(self, axis, value):
        """
        data set splitting on a given feature
        :param axis: the feature we'll split on
        :param value: the value of the feature
        :return:
        """
        ret_dataset = []
        for feature_vec in self.__dataset:
            if feature_vec[axis] == value:
                reduce_feature_vec = feature_vec[:axis]
                reduce_feature_vec.extend(feature_vec[axis + 1:])
                ret_dataset.append(reduce_feature_vec)
        return ret_dataset

    def __choose_best_feature_to_split(self):
        """
        choose best method to split dataset
        :return: best feature index
        """
        num_features = len(self.__dataset[0]) - 1  # the last column is used for the labels
        base_entropy = calc_shannon_ent(self.__dataset)
        best_info_grain = 0.0
        best_feature = -1
        # iterate over all the features
        for i in range(num_features):
            # create a set of all the examples of this feature
            feature_list = [ex[i] for ex in self.__dataset]
            unique_values = set(feature_list)
            new_entropy = 0.0
            for value in unique_values:
                sub_dataset = self.__split_dataset(i, value)
                prob = len(sub_dataset) / float(len(self.__dataset))
                new_entropy += prob * calc_shannon_ent(sub_dataset)
            # calculate the info gain, i.e., reduction in entropy
            info_gain = base_entropy - new_entropy
            # compare this to the best gain, if better than current best, set to best
            if info_gain > best_info_grain:
                best_info_grain = info_gain
                best_feature = i
        return best_feature


if __name__ == "__main__":
    simple_test = SimpleClassify()
    simple_test.test()
