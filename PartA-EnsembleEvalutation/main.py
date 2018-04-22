import numpy as np
import pandas as pd
from datasets.BankCustomerDataset import BankCustomerDataset
from datasets.CarEvaluationDataset import CarEvaluationDataset
from datasets.ImageSegmentationDataset import ImageSegmentationDataset
from datasets.IncomeDataset import IncomeDataset
from datasets.LetterRecognitionDataset import LetterRecognitionDataset
from datasets.MammMassDataset import MammMassDataset
from datasets.MushroomDataset import MushroomDataset
from datasets.TicTacToeDataset import TicTacToeDataset
from datasets.WineQualityDataset import WineQualityDataset
from datasets.YeastDataset import YeastDataset
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

import os
if not os.path.exists("datasets"):
    os.chdir("..")

datasets = {
    "Yeast": YeastDataset("datasets/files/yeast.data"),
    "Car Evaluation": CarEvaluationDataset("datasets/files/car.data"),
    "Letter Recognition Dataset": LetterRecognitionDataset("datasets/files/letter-recognition.data"),
    "Image Segmantation Dataset": ImageSegmentationDataset("datasets/files/segmentation.data"),
    'Wine Quality': WineQualityDataset('datasets/files/winequality'),
    'Income Evaluation': IncomeDataset('datasets/files/income.data'),
    'Bank Customer': BankCustomerDataset('datasets/files/bank-additional.csv'),
    'Mammographic Masses': MammMassDataset('datasets/files/mammographic-masses.data'),
    'Mushroom': MushroomDataset('datasets/files/mushroom.data'),
    'Tic Tac Toe': TicTacToeDataset('datasets/files/tic-tac-toe.data')
}

models = {
    'Bagging Classifier': BaggingClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'AdaBoost Classifier binary': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    #'AdaBoost Classifier multiclass': AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600, learning_rate=1.5, algorithm="SAMME"),
    #'AdaBoost Classifier': AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=100, random_state=1),
    'Gradient Boosting Classifier': GradientBoostingClassifier()
}

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro'
}

final_results = pd.DataFrame(index=datasets.keys(), columns=models.keys())
results = {}


def calculate_ranks(final_results):
    ranks = final_results.rank(ascending=False, axis=1)
    average_ranks = ranks.mean(axis=0)

    return average_ranks


# Train each model for each dataset on a N-Fold Cross Validation
for dataset_name in datasets:

    dataset = datasets[dataset_name]
    dataset.preprocessing()

    model_results = {}

    for model_name in models:
        model = models[model_name]

        # F1-Macro for now, will add more later on
        scores = cross_validate(X=dataset.get_x(), y=dataset.get_y(), estimator=model, scoring=scoring, cv=5)

        model_results[model_name] = {}
        for score in scoring:
            model_results[model_name][score] = np.mean(scores['test_'+score])

    results[dataset_name] = model_results

print('Results:')
for dataset_name in datasets:
    print('Dataset: {}'.format(dataset_name))

    for model_name in models:
        print('\tModel: {}'.format(model_name))

        for metric in results[dataset_name][model_name]:
            print('\t\t{}: {}'.format(metric, results[dataset_name][model_name][metric]))
        final_results.loc[dataset_name, model_name] = results[dataset_name][model_name]['accuracy']

average_ranks = calculate_ranks(final_results)
print('\nThe average ranks of the algorithms are:\n {}'.format(average_ranks))
