import numpy as np
import pandas as pd
from datasets.Dataset import Dataset


class MammMassDataset(Dataset):

    def __init__(self,data_file):
        super(MammMassDataset,self).__init__()
        dataset = pd.read_csv(data_file, delimiter=r',', header=None)
        dataset.columns = ['BI_RADS_assesment', 'Age', 'Shape', 'Margin', 'Density', 'y']
        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]


        """
        PREPROCESSING
        We remove the NaN values,
        we apply one-hot encoding to the columns that is necessary 
        and we normalize our data in the interval [0,1].
        """
    def preprocessing(self):

        self.removeNA()
        expl_vars = self.get_x()

        one_hot_list = ['Shape', 'Margin']
        for attr in one_hot_list:
            expl_vars = self.one_hot_encoding(expl_vars,attr)

        norm_expl_vars = self.normalize(expl_vars)
        self._x = pd.DataFrame(data=norm_expl_vars)

    def removeNA(self):
        dataset = pd.concat([self.get_x(), self.get_y()], axis=1)
        dataset = dataset.replace('?', np.nan)
        dataset = dataset.dropna(axis=0, how='any')
        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]
