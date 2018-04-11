from datasets.Dataset import Dataset
import pandas as pd


class WineQualityDataset(Dataset):

    def __init__(self, data_file):
        super(WineQualityDataset, self).__init__()

        # Loading the 2 files with the red and the white wines.
        red_wines = pd.read_csv((data_file + '-red.csv'), delimiter=';', header=0)
        white_wines = pd.read_csv(data_file + '-white.csv', delimiter=';', header=0)

        # Concat the two arrays in one.
        frames = [red_wines, white_wines]
        dataset = pd.concat(frames)

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

        norm_expl_vars = self.normalize(expl_vars)
        self._x = pd.DataFrame(data=norm_expl_vars)

