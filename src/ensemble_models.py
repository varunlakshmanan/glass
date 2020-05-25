from itertools import combinations
from varname import nameof
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import roc_auc_score


def ensemble_models(optimized_estimators, x_train, y_train, x_test, y_test, is_classifier):
    MAX_ENSEMBLES = 4
    all_estimator_combinations = []

    # Store combinations of length 2 to length MAX_ENSEMBLES of all estimators in a list
    for i in reversed(range(2, MAX_ENSEMBLES)):
        all_estimator_combinations = combinations(optimized_estimators.values(), i)
        all_estimator_combinations.append(all_estimator_combinations)

    voting_estimators = []

    if is_classifier:
        # Convert each element of all_estimator_combinations to a VotingClassifier() and store that VotingClassifier()
        # in the voting_estimators list
        for estimator_combination in all_estimator_combinations:
            names = []
            for estimator in estimator_combination:
                names.append(nameof(estimator))
            estimator_dict = dict(zip(names, estimator_combination))
            voting_estimators.append(VotingClassifier(estimator_dict))

        best_estimator = VotingClassifier()

        # Find the voting classifier with the highest AUC
        for voting_estimator in voting_estimators:
            max_auc = -1

            voting_estimator.fit(x_train, y_train)
            y_predictions = voting_estimator.predict(x_test)
            auc = roc_auc_score(y_test, y_predictions)

            if auc > max_auc:
                best_estimator = voting_estimator

        return best_estimator

    else:
        # Convert each element of all_estimator_combinations to a VotingRegressor() and store that VotingRegressor()
        # in the voting_estimators list
        for estimator_combination in all_estimator_combinations:
            names = []
            for estimator in estimator_combination:
                names.append(nameof(estimator))
            estimator_dict = dict(zip(names, estimator_combination))
            voting_estimators.append(VotingRegressor(estimator_dict))

        best_estimator = VotingRegressor()

        # Find the voting regressor with the highest AUC
        for voting_estimator in voting_estimators:
            max_auc = -1

            voting_estimator.fit(x_train, y_train)
            y_predictions = voting_estimator.predict(x_test)
            auc = roc_auc_score(y_test, y_predictions)

            if auc > max_auc:
                best_estimator = voting_estimator

        return best_estimator


def describe_ensemble():
    print(1)
