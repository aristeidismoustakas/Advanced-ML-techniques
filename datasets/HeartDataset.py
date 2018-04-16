from datasets.Dataset import Dataset
import pandas as pd


class HeartDataset(Dataset):

    def __init__(self, data_file):
        super(HeartDataset, self).__init__()

        dataset = pd.read_csv(data_file, delimiter=' ', header=0)

        dataset.columns = ['age', 'sex', 'chest_pain_type', 'blood pressure', 'serum cholestora',
                           'fasting_blood_sugar', 'electrocardiographic_results', 'maximum_heart_rate',
                           'exercise_induced_angina', 'oldpeak', 'slope_of_the_peak_exercise', 'number_of_major_vessels', 'thal',
                           'absence_or_presence']
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
        expl_vars = self.get_x()
        one_hot_list = ['sex', 'fasting_blood_sugar', 'exercise_induced_angina', 'chest_pain_type', 'electrocardiographic_results', 'thal']  #

        for attr in one_hot_list:
            expl_vars = self.one_hot_encoding(expl_vars, attr)

        norm_expl_vars = self.normalize(expl_vars)
        self._x = pd.DataFrame(data=norm_expl_vars)

