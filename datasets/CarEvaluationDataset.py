from datasets.Dataset import Dataset
from sklearn import preprocessing
import pandas as pd

# https://archive.ics.uci.edu/ml/machine-learning-databases/car/

class CarEvaluationDataset(Dataset):

    def __init__(self, data_file):
        super(CarEvaluationDataset, self).__init__()

        df = pd.read_csv(data_file, delimiter=",")
        df.columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

        door_mapping = {"1": 0, "2": 1, "3": 2, "4": 3, "5more": 4}
        person_mapping = {"2": 0, "4": 1, "more": 2}

        df["doors"] = df["doors"].apply(lambda x: door_mapping[x])
        df["persons"] = df["persons"].apply(lambda x: person_mapping[x])

        for col in df.columns:
            if col not in ["door", "persons"]:
                l_enc = preprocessing.LabelEncoder()
                df[col] = l_enc.fit_transform(df[col])

        self._x = df.iloc[:, :-1]
        self._y = df.iloc[:, -1]
