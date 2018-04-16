from sklearn.model_selection import cross_validate
from datasets.HeartDataset import HeartDataset


dataset = HeartDataset('datasets/files/heart.dat')
dataset.preprocessing()

techniques = {

}

algorithms = {

}

scoring = {
    'accuracy': 'accuracy',
    'f1_macro': 'f1_macro'
}

results = {}

# Train each algorithm using each technique on a N-Fold Cross Validation
for technique_name in techniques:

    technique = techniques[technique_name]

    algorithm_results = {}

    for algorithm_name in algorithms:
        model = algorithms[algorithm_name]

        # F1-Macro for now, will add more later on
        scores = cross_validate(X=dataset.get_x(), y=dataset.get_y(), estimator=model, scoring=scoring, cv=5)

        algorithm_results[algorithm_name] = {}
        for score in scoring:
            algorithm_results[algorithm_name][score] = np.mean(scores['test_'+score])

    results[technique_name] = algorithm_results

print('Results:')
for technique_name in techniques:
    print('Technique: {}'.format(technique_name))

    for algorithm_name in algorithms:
        print('\tAlgorithm: {}'.format(algorithm_name))

        for metric in results[technique_name][algorithm_name]:
            print('\t\t{}: {}'.format(metric, results[technique_name][algorithm_name][metric]))