import optuna
from sklearn.model_selection import cross_val_score
from varname import nameof

def dt_objective(trial, x, y, estimator):
    max_depth = int(trial.suggest_discrete_uniform("max_depth", 1, 500, 1))
    min_samples_split = trial.suggest_discrete_uniform("min_samples_split", 0.005, 0.5, 0.005)
    min_samples_leaf = trial.suggest_discrete_uniform("min_samples_leaf", 0.001, 0.1, 0.001)
    min_weight_fraction_leaf = trial.suggest_discrete_uniform("min_weight_fraction_leaf", 0.0, 1.0, 0.01)
    max_features = trial.suggest_discrete_uniform("max_features", 0.01, 0.25, 0.01)
    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 2, 512, 1))
    params = {
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_features": max_features,
        "max_leaf_nodes": max_leaf_nodes
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def rf_objective(trial, x, y, estimator):
    n_estimators = int(trial.suggest_discrete_uniform("n_estimators", 20, 4000, 1))
    max_depth = int(trial.suggest_discrete_uniform("max_depth", 1, 500, 1))
    min_samples_split = trial.suggest_discrete_uniform("min_samples_split", 0.005, 0.5, 0.005)
    min_samples_leaf = trial.suggest_discrete_uniform("min_samples_leaf", 0.001, 0.1, 0.001)
    min_weight_fraction_leaf = trial.suggest_discrete_uniform("min_weight_fraction_leaf", 0.0, 1.0, 0.01)
    max_features = trial.suggest_discrete_uniform("max_features", 0.01, 0.25, 0.01)
    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 2, 512, 1))
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_features": max_features,
        "max_leaf_nodes": max_leaf_nodes
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def ab_objective(trial, x, y, estimator):
    n_estimators = int(trial.suggest_discrete_uniform("n_estimators", 10, 1000, 1))
    learning_rate = trial.suggest_discrete_uniform("learning_rate", 0.001, 1, 0.0001)
    params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def bag_objective(trial, x, y, estimator):
    n_estimators = int(trial.suggest_discrete_uniform("n_estimators", 10, 1000, 1))
    max_samples = trial.suggest_discrete_uniform(0.0, 1.0, 0.01)
    max_features = trial.suggest_discrete_uniform("max_features", 0.01, 0.25, 0.01)
    params = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "max_features": max_features
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def et_objective(trial, x, y, estimator):
    n_estimators = int(trial.suggest_discrete_uniform("n_estimators", 20, 4000, 1))
    max_depth = int(trial.suggest_discrete_uniform("max_depth", 1, 500, 1))
    min_samples_split = trial.suggest_discrete_uniform("min_samples_split", 0.005, 0.5, 0.005)
    min_samples_leaf = trial.suggest_discrete_uniform("min_samples_leaf", 0.001, 0.1, 0.001)
    min_weight_fraction_leaf = trial.suggest_discrete_uniform("min_weight_fraction_leaf", 0.0, 1.0, 0.01)
    max_features = trial.suggest_discrete_uniform("max_features", 0.01, 0.25, 0.01)
    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 2, 512, 1))
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_features": max_features,
        "max_leaf_nodes": max_leaf_nodes
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def gb_objective(trial, x, y, estimator):
    loss = trial.suggest_categorical("loss", "ls", "lad", "huber")
    learning_rate = int(trial.suggest_discrete_uniform("learning_rate", 0.001, 1, 0.0001))
    n_estimators = int(trial.suggest_discrete_uniform("n_estimators", 20, 4000, 1))
    min_samples_split = int(trial.suggest_discrete_uniform("min_samples_split", 0.005, 0.5, 0.005))
    min_samples_leaf = int(trial.suggest_discrete_uniform("min_samples_leaf", 0.001, 0.1, 0.001))
    min_weight_fraction_leaf = int(trial.suggest_discrete_uniform("min_weight_fraction_leaf", 0.0, 1.0, 0.01))
    max_depth = int(trial.suggest_discrete_uniform("max_depth", 1, 500, 1))
    max_features = int(trial.suggest_discrete_uniform("max_features", 0.01, 0.25, 0.01))
    max_leaf_nodes = int(trial.suggest_discrete_uniform("max_leaf_nodes", 2, 512, 1))
    params = {
        "loss": loss,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": min_weight_fraction_leaf,
        "max_depth": max_depth,
        "max_features": max_features,
        "max_leaf_nodes": max_leaf_nodes
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def gpc_objective(trial, x, y, estimator):
    n_restarts_optimizer = int(trial.suggest_discrete_uniform("n_restarts_optimizer", 0, 30, 1))
    params = {
        "n_restarts_optimizer": n_restarts_optimizer
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def lr_objective(trial, x, y, estimator):
    penalty = trial.suggest_categorical("penalty", "l2", "none")
    tol = trial.suggest_discrete_uniform("tol", 10**(-6), 10**(-2), 10**(-4))
    c = trial.suggest_discrete_uniform("C", 0.01, 10, 0.1)
    solver = trial.suggest_categorical("solver", "newton-cg", "lbfgs", "sag", "saga", "liblinear")
    params = {
        "penalty": penalty,
        "tol": tol,
        "C": c,
        "solver": solver,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def pa_objective(trial, x, y, estimator):
    c = trial.suggest_discrete_uniform("C", 0.01, 10, 0.1)
    max_iter = int(trial.suggest_discrete_uniform("max_iter", 500, 2000, 1))
    tol = trial.suggest_discrete_uniform("tol", 10**(-5), 10**(-1), 10**(-3))
    params = {
        "C": c,
        "max_iter": max_iter,
        "tol": tol,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def ridge_objective(trial, x, y, estimator):
    alpha = trial.suggest_discrete_uniform("alpha", 0.01, 10, 0.1)
    max_iter = int(trial.suggest_discrete_uniform("max_iter", 500, 2000, 1))
    tol = trial.suggest_discrete_uniform("tol", 10**(-5), 10**(-1), 10**(-3))
    solver = trial.suggest_categorical("solver", "auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga")
    params = {
        "alpha": alpha,
        "max_iter": max_iter,
        "tol": tol,
        "solver": solver
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def sgd_objective(trial, x, y, estimator):
    penalty = trial.suggest_categorical("penalty", "l2", "l1", "elasticnet")
    alpha = trial.suggest_discrete_uniform("alpha", 0.00001, 0.01, 0.001)
    l1_ratio = trial.suggest_discrete_uniform("l1_ratio", 0.01, 0.3, 0.01)
    max_iter = int(trial.suggest_discrete_uniform("max_iter", 500, 2000, 1))
    tol = trial.suggest_discrete_uniform("tol", 10**(-5), 10**(-1), 10**(-3))
    params = {
        "penalty": penalty,
        "alpha": alpha,
        "l1_ratio": l1_ratio,
        "max_iter": max_iter,
        "tol": tol,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def per_objective(trial, x, y, estimator):
    penalty = trial.suggest_categorical("penalty", "l2", "l1", "elasticnet")
    alpha = trial.suggest_discrete_uniform("alpha", 0.00001, 0.01, 0.001)
    max_iter = int(trial.suggest_discrete_uniform("max_iter", 500, 2000, 1))
    tol = trial.suggest_discrete_uniform("tol", 10**(-5), 10**(-1), 10**(-3))
    params = {
        "penalty": penalty,
        "alpha": alpha,
        "max_iter": max_iter,
        "tol": tol,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def sv_objective(trial, x, y, estimator):
    c = trial.suggest_discrete_uniform("C", 0.01, 10, 0.1)
    tol = trial.suggest_discrete_uniform("tol", 10**(-5), 10**(-1), 10**(-3))
    params = {
        "C": c,
        "tol": tol,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def nu_sv_objective(trial, x, y, estimator):
    nu = trial.suggest_discrete_uniform("C", 0, 1, 0.01)
    tol = trial.suggest_discrete_uniform("tol", 10**(-5), 10**(-1), 10**(-3))
    params = {
        "nu": nu,
        "tol": tol,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def l_sv_objective(trial, x, y, estimator):
    c = trial.suggest_discrete_uniform("C", 0.01, 10, 0.1)
    tol = trial.suggest_discrete_uniform("tol", 10**(-5), 10**(-1), 10**(-3))
    max_iter = int(trial.suggest_discrete_uniform("max_iter", 500, 2000, 1))
    params = {
        "c": c,
        "tol": tol,
        "max_iter": max_iter
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def kn_objective(trial, x, y, estimator):
    n_neighbors = int(trial.suggest_discrete_uniform("n_neighbors", 1, 25, 1))
    weights = trial.suggest_categorical("weights", "uniform", "distance")
    algorithm = trial.suggest_categorical("algorithm", "auto", "ball_tree", "kd_tree", "brute")
    p = int(trial.suggest_discrete_uniform("max_iter", 1, 10, 1))
    params = {
        "n_neighbors": n_neighbors,
        "weights": weights,
        "algorithm": algorithm,
        "p": p
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def bnb_objective(trial, x, y, estimator):
    alpha = trial.suggest_discrete_uniform("alpha", 0.0, 2.0, 0.001)
    binarize = trial.suggest_discrete_uniform("binarize", 0.0, 2.0, 0.001)
    params = {
        "alpha": alpha,
        "binarize": binarize,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def gnb_objective(trial, x, y, estimator):
    var_smoothing = trial.suggest_discrete_uniform("var_smoothing", 1**(-10), 1**(-8), 1**(-9))
    params = {
        "var_smoothing": var_smoothing,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def lda_objective(trial, x, y, estimator):
    solver = trial.suggest_categorical("solver", "svd", "lsqr", "eigen")
    params = {
        "solver": solver,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def qda_objective(trial, x, y, estimator):
    reg_param = trial.suggest_discrete_uniform(0.0, 1.0, 0.001)
    params = {
        "reg_param": reg_param,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def xgb_objective(trial, x, y, estimator):
    n_estimators = int(trial.suggest_discrete_uniform(50, 1500, 1))
    max_depth = int(trial.suggest_discrete_uniform(2, 2056, 1))
    learning_rate = trial.suggest_discrete_uniform(0.0, 1.0, 0.001)
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def lgbm_objective(trial, x, y, estimator):
    boosting_type = trial.suggest_categorical("boosting_type", "gbdt", "dart", "goss", "rf")
    num_leaves = int(trial.suggest_discrete_uniform("num_leaves", 15, 511, 1))
    max_depth = int(trial.suggest_discrete_uniform("max_depth", -1, 16, 1))
    learning_rate = trial.suggest_discrete_uniform("learning_rate", 0.001, 0.25, 0.001)
    n_estimators = int(trial.suggest_discrete_uniform("n_estimators", 20, 4000, 1))
    params = {
        "boosting_type": boosting_type,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def optimize_hyperparams(estimators, x, y):
    optimized_estimators = []

    for estimator in estimators:
        if nameof(estimator) == "decision_tree":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: dt_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "random_forest":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: rf_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "adaboost":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: ab_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "bagging":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: bag_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "extra_trees":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: et_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "gradient_boosting":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: gb_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "gaussian_process_classifier":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: gpc_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "logistic_regression":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: lr_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "passive_aggressive":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: pa_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "ridge":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: ridge_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "sgd":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: sgd_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "perceptron":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: per_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "svc" or nameof(estimator) == "svr":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: sv_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "nu_svc" or nameof(estimator) == "nu_svr":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: nu_sv_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "linear_svc" or nameof(estimator) == "linear_svr":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: l_sv_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "k_neighbors":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: kn_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "bernoulli_nb":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: bnb_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "gaussian_nb":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: gnb_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "linear_discriminant_analysis":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: lda_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "quadratic_discriminant_analysis":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: qda_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "xgb":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: xgb_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)
        elif nameof(estimator) == "lgbm":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: lgbm_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators.append(estimator)

    return optimized_estimators
