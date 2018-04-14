from datasets.Dataset import Dataset
import pandas as pd

class TicTacToeDataset(Dataset):

    def __init__(self,data_file):
        super(TicTacToeDataset,self).__init__()
        dataset = pd.read_csv(data_file, delimiter=r',', header=None)
        dataset.columns = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square',
                           'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'y']
        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]


    """
        PREPROCESSING
        We apply one-hot encoding to the columns that is necessary 
        and we normalize our data in the interval [0,1].
        We don't have missing values in this dataset.
    """

    def preprocessing(self):
        expl_vars = self.get_x()

        one_hot_list = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square',
                           'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square']
        for attr in one_hot_list:
            expl_vars = self.one_hot_encoding(expl_vars, attr)

        norm_expl_vars = self.normalize(expl_vars)
        self._x = pd.DataFrame(data=norm_expl_vars)
