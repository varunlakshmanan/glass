from fit_models import fit_models
from optimize_hyperparams import optimize_hyperparams
from ensemble_models import ensemble_models

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import roc_auc_score

ensemble = VotingClassifier()


class GlassClassifier:
    def __init__(self):
        pass

    def fit(self, x, y):
        is_classifier = True
        global ensemble
        ensemble = ensemble_models(optimize_hyperparams(fit_models(x, y, is_classifier)))

    def predict(self, x):
        y_predictions = ensemble.predict(x)
        return y_predictions

    def describe(self, y_test, y_preds):
        auc = roc_auc_score(y_test, y_preds)
        print("AUC: " + auc)
