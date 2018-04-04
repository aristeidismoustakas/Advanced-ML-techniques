class Dataset(object):

    def __init__(self):
        self._x = []
        self._y = []

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def discretize(self, data):
        """
        Discretizes an array of data.
        First we count the number of different elements,
        then we assign a number to each element and add it to
        a new list, which we finally return.
        :param data: Data to discretize (1-D array of elements)
        :return: Discretized data
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