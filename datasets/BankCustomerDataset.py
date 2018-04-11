from datasets.Dataset import Dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing


class BankCustomerDataset(Dataset):

    def __init__(self, data_file):
        super(BankCustomerDataset, self).__init__()

        dataset = pd.read_csv(data_file, delimiter=';', header=0)
        dataset.columns = ['age', 'job', 'marital', 'education', 'default', 'housing',
                            'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign',
                            'pdays', 'previous', 'poutcome', 'emp_var_rate', 'cons_price_idx',
                            'cons_conf_idx', 'euribor3m', 'nr_employed', 'y']
        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]

    def preprocessing(self):
        """
        Preproccesing of the explanatory variables.
        At first we remove the NA values, next we
        apply one-hot encoding to the columns that is
        necessary and finally we normalize our data in
        the interval 0-1.
        """
        self.removeNA()
        expl_vars = self.get_x()
        one_hot_list = ['job', 'marital', 'education', 'default', 'housing',
                        'loan', 'contact', 'month', 'day_of_week', 'poutcome'] #

        for attr in one_hot_list:
            expl_vars = self.one_hot_encoding(expl_vars, attr)

        norm_expl_vars = self.normalize(expl_vars)
        self._x = pd.DataFrame(data=norm_expl_vars)


    def removeNA(self):
        """
        Elimination of the examples
        who contain at least one NA value
        in their attributes.
        """
        dataset = pd.concat([self.get_x(), self.get_y()], axis=1)
        dataset = dataset.replace('unknown', np.nan)
        dataset = dataset.dropna(axis=0, how='any')
        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]
