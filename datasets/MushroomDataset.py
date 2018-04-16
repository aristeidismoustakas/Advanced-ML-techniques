from datasets.Dataset import Dataset
import pandas as pd
import numpy as np

class MushroomDataset(Dataset):

    def __init__(self,data_file):
        super(MushroomDataset,self).__init__()
        dataset = pd.read_csv(data_file, delimiter=r',', header=None)
        dataset.columns = ['y', 'cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
                           'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring',
                           'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
        self._x = dataset.iloc[:, 1:]
        self._y = dataset.iloc[:, 0]

        """
            PREPROCESSING
            We remove the NaN values,
            we apply one-hot encoding to the columns that is necessary 
            and we normalize our data in the interval [0,1].
        """

    def preprocessing(self):
        self.removeNA()
        expl_vars = self.get_x()

        one_hot_list = ['cap_shape', 'cap_surface', 'cap_color', 'bruises', 'odor', 'gill_attachment', 'gill_spacing', 'gill_size', 'gill_color',
                           'stalk_shape', 'stalk_root', 'stalk_surface_above_ring', 'stalk_surface_below_ring', 'stalk_color_above_ring', 'stalk_color_below_ring',
                           'veil_type', 'veil_color', 'ring_number', 'ring_type', 'spore_print_color', 'population', 'habitat']
        for attr in one_hot_list:
            expl_vars = self.one_hot_encoding(expl_vars,attr)

        norm_expl_vars = self.normalize(expl_vars)
        self._x = pd.DataFrame(data=norm_expl_vars)

    def removeNA(self):
        dataset = pd.concat([self.get_x(), self.get_y()], axis=1)
        dataset = dataset.replace('?', np.nan)
        dataset = dataset.dropna(axis=0, how='any')
        self._x = dataset.iloc[:, 1:]
        self._y = dataset.iloc[:, 0]
