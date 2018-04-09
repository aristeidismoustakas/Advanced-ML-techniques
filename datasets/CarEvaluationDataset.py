from datasets.Dataset import Dataset

class CarEvaluationDataset(Dataset):

    def __init__(self, data_file):
        super(CarEvaluationDataset, self).__init__()

        lines = []
        with open(data_file, "r") as f:
            lines = f.readlines()

        x_columns = []

        for line in lines:
            if line.endswith("\n"):
                line = line[:-1]

            data = line.split(",")

            if x_columns == []:
                for i in range(len(data[:-1])):
                    x_columns.append([])

            for index, val in enumerate(data[:-1]):
                x_columns[index].append(val)

            self._y.append(data[-1])

        # Discretize Y
        self._y = self.enumerate(self._y)

        # Discretize X columns
        for i in range(len(x_columns)):
            x_columns[i] = self.enumerate(x_columns[i])

        # Join X columns in rows
        for i in range(len(self._y)):
            x_row = []
            for j in range(len(x_columns)):
                x_row.append(x_columns[j][i])

            self._x.append(x_row)

