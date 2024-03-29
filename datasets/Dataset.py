import pandas as pd
from sklearn import preprocessing

class Dataset(object):

    def __init__(self):
        self._x = []
        self._y = []

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def preprocessing(self):
        pass

    def removeNA(self):
        pass

    def normalize(self, data):
        """
        Normalizes an array of data in the interval 0-1
        :param data: Data to normalize (2-D array of elements)
        :return: Normalized data
        """
        new_data = preprocessing.normalize(data, norm='max', axis=0)
        return new_data

    def one_hot_encoding(self, data, col, weight=1):
        """
        One-hot encoding for a column of an array.
        We want to apply the one-hot encoding to a
        specific column of a 2-D array. So, we take the
        distinct values of this column and we create a
        new column for each of them distinct values.
        Next, we put 1 only in one of these columns for each entry
        and 0 to all the others. In addition, we can use a weight
        in order not to be favored some attributes because of the
        calculation of distances of some algorithms.
        :param data: Array of data (2-D array of elements)
        :param col: The column for the one-hot encoding
        :param weight: Weight for the mitigation of
        the effect of the 0-1 values in the calculation
        of distances.
        :return: 2-D array with the new columns
        """
        one_hot = pd.get_dummies(data[col])
        one_hot.columns = [col + str(column) for column in one_hot.columns]
        data = data.drop(col, axis=1)
        data = data * weight
        new_data = data.join(one_hot)

        return new_data

    def enumerate(self, data):
        """
        Enumerates an array of data.
        First we count the number of different elements,
        then we assign a number to each element and add it to
        a new list, which we finally return.
        :param data: Data to enumerate (1-D array of elements)
        :return: Enumerated data
        """

        diff_values = set()

        for val in data:
            if val not in diff_values:
                diff_values.add(val)

        value_to_number_map = {}
        for val in diff_values:
            value_to_number_map[val] = len(value_to_number_map)

        new_data = []
        for val in data:
            new_data.append(value_to_number_map[val])

        return new_data