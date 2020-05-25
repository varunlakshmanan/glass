from build_models import build_models
from optimize_hyperparams import optimize_hyperparams
from ensemble_models import ensemble_models

global ensemble
auc = -1


class GlassRegressor:
    def __init__(self):
        pass

    def fit(self, x_train, y_train, x_test, y_test):
        is_classifier = False
        global ensemble
        ensemble = ensemble_models(optimize_hyperparams(build_models(is_classifier), x_train, y_train),
                                   x_train, y_train, x_test, y_test, is_classifier)

    def predict(self, x_test):
        global auc
        y_predictions, auc = ensemble.predict(x_test)
        return y_predictions

    def describe(self):
        print("Highest AUC: " + auc)
        print(ensemble.get_params(self, deep=True))
