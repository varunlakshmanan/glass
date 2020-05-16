from fit_models import fit_models
from optimize_hyperparams import optimize_hyperparams
from ensemble_models import ensemble_models

class GlassClassifier():
    def __init__(self):
        pass

    def build_ensemble(self, input_data):
        is_classifier = True
        fit_models(input_data, is_classifier)

    def properties(self, input_data):
            print(1)
