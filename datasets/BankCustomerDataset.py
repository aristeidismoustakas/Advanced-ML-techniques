from datasets.Dataset import Dataset
import pandas as pd
from sklearn import preprocessing


class BankCustomerDataset(Dataset):

    def __init__(self, data_file):
        super(BankCustomerDataset, self).__init__()

        dataset = pd.read_csv(data_file, delimiter=';', header=0)

        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]

    def preprocessing(self):
        expl_vars = self.get_x()
        expl_vars[['age']] = expl_vars[['age']] + 100
        self._x = expl_vars
