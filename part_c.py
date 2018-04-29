import numpy as np
from techniques.Base import Base
from techniques.EasyEnsembleTechnique import EasyEnsembleTechnique
from techniques.SMOTETechnique import SMOTETechnique
from techniques.NearMissTechnique import NearMissTechnique
from imblearn.metrics import geometric_mean_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score

from datasets.CreditCardFraudDataset import CreditCardFraudDataset


dataset = CreditCardFraudDataset('datasets/files/creditcard.csv')
dataset.preprocessing()


def cross_validate(model, x, y, cv=5):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    results = {
        "recall": [],
        "accuracy": [],
        "f1": [],
        "geometric-gmean": [],
        "roc_auc_score": []
    }

    for train_index, test_index in kf.split(x):
        fit = model.fit([x[index] for index in train_index], [y[index] for index in train_index])

        predictions = fit.predict([x[index] for index in test_index])
        y_true = [y[index] for index in test_index]

        results["recall"].append(recall_score(y_true, predictions))
        results["accuracy"].append(accuracy_score(y_true, predictions))
        results["f1"].append(f1_score(y_true, predictions))
        results["geometric-gmean"].append(geometric_mean_score(y_true, predictions, average='weighted'))
        results["roc_auc_score"].append(roc_auc_score(y_true, predictions))

    return results

# Techniques format:
# Class name and named arguments (model will be given automatically)
techniques = {
    "Base": (Base, {}),
    # "EasyEnsemble": (EasyEnsembleTechnique, {"n_estimators": 10}),
    "SMOTETechnique": (SMOTETechnique, {}),
    # "NearMiss1": (NearMissTechnique, {"version": 1}),
    # "NearMiss2": (NearMissTechnique, {"version": 2}),
    # "NearMiss3": (NearMissTechnique, {"version": 3})

}

models = {
    "Naive Bayes": GaussianNB(),
    "Linear SVC": LinearSVC(),
    "Random Forests": RandomForestClassifier(),
}

#
# scoring = {
#     'Log Loss': 'neg_log_loss'
# }

results = {}

# Train each algorithm using each technique on a N-Fold Cross Validation
for technique_name in techniques:

    technique_class = techniques[technique_name][0]
    technique_kwargs = techniques[technique_name][1]

    model_scores = {}

    for model_name in models:
        model = models[model_name]
        technique_estimator = technique_class(model, **technique_kwargs)

        scores = cross_validate(
            technique_estimator,
            np.asarray(dataset.get_x()),
            np.asarray(dataset.get_y()),
            cv=5)

        model_scores[model_name] = {}
        for score in scores:
            model_scores[model_name][score] = np.mean(scores[score])

    results[technique_name] = model_scores

print('Results:')
for technique_name in techniques:
    print('Technique: {}'.format(technique_name))

    for model_name in models:
        print('\tAlgorithm: {}'.format(model_name))

        for metric in results[technique_name][model_name]:
            print('\t\t{}: {}'.format(metric, results[technique_name][model_name][metric]))