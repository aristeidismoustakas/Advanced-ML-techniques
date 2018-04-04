from Dataset import Dataset

class YeastDataset(Dataset):

    def __init__(self, data_file):
        super(YeastDataset, self).__init__()

        lines = []
        with open(data_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Remove \n
            if line.endswith("\n"):
                line = line[:-1]

            data = line.split()

            # First attribute is label, ignore
            self._x.append([float(val) for val in data[1:-1]])

            self._y.append(data[-1])

        self._y = self.discretize(self._y)
