import pandas as pd
from datasets.Dataset import Dataset


# https://archive.ics.uci.edu/ml/datasets/Letter+Recognition

class LetterRecognitionDataset(Dataset):

    def __init__(self, data_file):
        super(LetterRecognitionDataset, self).__init__()

        df = pd.read_csv(data_file, delimiter=",")

        df.columns = ["letter", "x-box", "y-box", "width", "height",
                      "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar",
                      "x2ybr", "xy2br", "x-edge", "xegvy", "y-ege", "yegvx"]

        for col in df.columns:
            if col != "letter":
                df[col] = pd.to_numeric(df[col])

        self._x = df.iloc[:, 1:]
        self._y = df.iloc[:, 0]
