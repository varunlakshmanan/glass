from fit_models import fit_models
from optimize_hyperparams import optimize_hyperparams
from ensemble_models import ensemble_models

from sklearn.ensemble import VotingRegressor

global ensemble
auc = -1


class GlassClassifier:
    def __init__(self):
        pass

    def fit(self, x_train, y_train, x_test, y_test):
        is_classifier = True
        global ensemble
        ensemble = ensemble_models(optimize_hyperparams(fit_models(is_classifier), x_train, y_train),
                                   x_train, y_train, x_test, y_test, is_classifier)

    def predict(self, x_test):
        global auc
        y_predictions, auc = ensemble.predict(x_test)
        return y_predictions

    def describe(self):
        print("Highest AUC: " + auc)
        print(ensemble.get_params(self, deep=True))