from datasets.Dataset import Dataset
import pandas as pd

#https://www.kaggle.com/mlg-ulb/creditcardfraud/data
#https://www.kaggle.com/mlg-ulb/creditcardfraud/version/3   #for description

class CreditCardFraudDataset(Dataset):

    def __init__(self,data_file):
        super(CreditCardFraudDataset,self).__init__()
        dataset = pd.read_csv(data_file, delimiter=',', header=0)
        dataset.columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18',
        'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'y']
        self._x = dataset.iloc[:, 0:-1]
        self._y = dataset.iloc[:, -1]

    """
        PREPROCESSING
        We normalize our numeric data in the interval [0,1].
    """

    def preprocessing(self):
        expl_vars = self.get_x()

        norm_expl_vars = self.normalize(expl_vars)
        self._x = pd.DataFrame(data=norm_expl_vars)
