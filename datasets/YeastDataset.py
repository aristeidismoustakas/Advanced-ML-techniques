from datasets.Dataset import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class YeastDataset(Dataset):

    def __init__(self, data_file):
        super(YeastDataset, self).__init__()

        df = pd.read_csv(data_file, delim_whitespace=True)
        df.columns = ["seq", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]
        df = df.drop("seq", 1)

        for col in df.columns:
            if col != "class":
                df[col] = pd.to_numeric(df[col])

        self._x = df.iloc[:, :-1]
        self._y = df.iloc[:, -1]