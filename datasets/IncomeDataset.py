from datasets.Dataset import Dataset
import pandas as pd


class IncomeDataset(Dataset):

    def __init__(self, data_file):
        super(IncomeDataset, self).__init__()

        dataset = pd.read_csv(data_file, delimiter=',', header=None)
        dataset.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education - num', 'marital - status', 'occupation', 'relationship', 'race', 'sex', 'capital - gain', 'capital - loss', 'hours - per - week', 'native - country', 'y']

        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]