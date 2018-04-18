import pandas as pd
from datasets.Dataset import Dataset


# https://archive.ics.uci.edu/ml/datasets/Image+Segmentation

class ImageSegmentationDataset(Dataset):

    def __init__(self, data_file):
        super(ImageSegmentationDataset, self).__init__()

        df = pd.read_csv(data_file, delimiter=",", header=5)

        df.columns = ["class", "region-centroid-col", "region-centroid-row",
                      "region-pixel-count", "short-line-density-5",
                      "short-line-density-2", "vedge-mean", "vegde-sd", "hedge-mean",
                      "hedge-sd ", "intensity-mean", "rawred-mean", "rawblue-mean",
                      "rawgreen-mean", "exred-mean", "exblue-mean", "exgreen-mean",
                      "value-mean", "saturatoin-mean", "hue-mean"
                      ]

        for col in df.columns:
            if col != "class":
                df[col] = pd.to_numeric(df[col])

        self._x = df.iloc[:, 1:]
        self._y = df.iloc[:, 0]
