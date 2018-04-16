from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_validate
from datasets.YeastDataset import YeastDataset
from datasets.CarEvaluationDataset import CarEvaluationDataset
from datasets.LetterRecognitionDataset import LetterRecognitionDataset
from datasets.ImageSegmentationDataset import ImageSegmentationDataset
from datasets.IncomeDataset import IncomeDataset
from datasets.WineQualityDataset import WineQualityDataset
from datasets.BankCustomerDataset import BankCustomerDataset

import numpy as np
import pandas as pd
from sklearn import preprocessing

datasets = {
    "Yeast": YeastDataset("datasets/files/yeast.data"),
    "Car Evaluation": CarEvaluationDataset("datasets/files/car.data"),
    "Letter Recognition Dataset": LetterRecognitionDataset("datasets/files/letter-recognition.data"),
    "Image Segmantation Dataset": ImageSegmentationDataset("datasets/files/segmentation.data"),
    'Wine Quality': WineQualityDataset('datasets/files/winequality'),
    'Income Evaluation': IncomeDataset('datasets/files/income.data'),
    'Bank Customer': BankCustomerDataset('datasets/files/bank-additional.csv')
}

models = {
    'Bagging Classifier': BaggingClassifier(),
    'Random Forest Classifier': RandomForestClassifier(),
    'AdaBoost Classifier binary': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
    'AdaBoost Classifier multiclass': AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=600, learning_rate=1.5, algorithm="SAMME")
}

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro'
}

results = {}

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