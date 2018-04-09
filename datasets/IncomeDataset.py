from datasets.Dataset import Dataset
import pandas as pd


class IncomeDataset(Dataset):

    def __init__(self, data_file):
        super(IncomeDataset, self).__init__()
        dataset = pd.read_csv(data_file, delimiter=',', header=None)
        dataset.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'y']
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
        one_hot_list = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
        for attr in one_hot_list:
            expl_vars = self.one_hot_encoding(expl_vars, attr)

        norm_expl_vars = self.normalize(expl_vars)
        self._x = pd.DataFrame(data = norm_expl_vars)

    def removeNA(self):
        """
        Elimination of the examples
        who contain at least one NA value
        in their attributes.
        """
        dataset = pd.concat([self.get_x(), self.get_y()], axis=1)
        dataset = dataset.dropna(axis=0, how='any')
        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]
