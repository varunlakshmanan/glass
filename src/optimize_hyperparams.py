import optuna
from sklearn.model_selection import cross_val_score
from varname import nameof


def dt_objective(trial, x, y, estimator):
    max_depth = int(trial.suggest_suggest_discrete_uniform("max_depth", 1, 500, 1))
    min_samples_split = int(trial.suggest_suggest_discrete_uniform("min_samples_split", 0.005, 0.5, 0.005))
    min_samples_leaf = int(trial.suggest_discrete_uniform("min_samples_leaf", 0.001, 0.1, 0.001))
    min_weight_fraction_leaf = int(trial.suggest_discrete_uniform("min_weight_fraction_leaf", 0.0, 1.0, 0.01))
    max_features = int(trial.suggest_discrete_uniform("max_features", 0.01, 0.25, 0.01))
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
    max_depth = int(trial.suggest_suggest_discrete_uniform("max_depth", 1, 500, 1))
    min_samples_split = int(trial.suggest_suggest_discrete_uniform("min_samples_split", 0.005, 0.5, 0.005))
    min_samples_leaf = int(trial.suggest_discrete_uniform("min_samples_leaf", 0.001, 0.1, 0.001))
    min_weight_fraction_leaf = int(trial.suggest_discrete_uniform("min_weight_fraction_leaf", 0.0, 1.0, 0.01))
    max_features = int(trial.suggest_discrete_uniform("max_features", 0.01, 0.25, 0.01))
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
    learning_rate = int(trial.suggest_discrete_uniform("learning_rate", 0.001, 1, 0.0001))
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
    max_samples = int(trial.suggest_discrete_uniform(0.0, 1.0, 0.01))
    max_features = int(trial.suggest_discrete_uniform("max_features", 0.01, 0.25, 0.01))
    params = {
        "n_estimators": n_estimators,
        "max_samples": max_samples,
        "max_features": max_features
    }
    estimator.set_params(**params)
    score = cross_val_score(estimator, x, y)
    accuracy = score.mean()
    return accuracy


def optimize_hyperparams(estimators, x, y):
    estimators_names = []
    for estimator in estimators:
        estimators_names.append(nameof(estimator))

    optimized_estimators = {}
    optimized_estimators.fromkeys(estimators_names)

    for estimator in estimators:
        if nameof(estimator) == "decision_tree":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: dt_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["decision_tree"] = estimator
        elif nameof(estimator) == "random_forest":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: rf_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["random_forest"] = estimator
        elif nameof(estimator) == "adaboost":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: ab_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["adaboost"] = estimator
        elif nameof(estimator) == "bagging":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: bag_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["bagging"] = estimator
        elif nameof(estimator) == "extra_trees":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: et_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["extra_trees"] = estimator
        elif nameof(estimator) == "gradient_boosting":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: gb_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["gradient_boosting"] = estimator
        elif nameof(estimator) == "gaussian_process_classifier":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: gpc_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["gaussian_process_classifier"] = estimator
        elif nameof(estimator) == "logistic_regression":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: lr_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["logistic_regression"] = estimator
        elif nameof(estimator) == "passive_aggressive":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: pa_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["passive_aggressive"] = estimator
        elif nameof(estimator) == "ridge":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: ridge_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["ridge"] = estimator
        elif nameof(estimator) == "sgd":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: sgd_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["sgd"] = estimator
        elif nameof(estimator) == "perceptron":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: per_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["perceptron"] = estimator
        elif nameof(estimator) == "svc":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: svc_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["svc"] = estimator
        elif nameof(estimator) == "nu_svc":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: nusvc_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["nu_svc"] = estimator
        elif nameof(estimator) == "linear_svc":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: lsvc_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["nu_svc"] = estimator
        elif nameof(estimator) == "k_neighbors":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: kn_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["k_neighbors"] = estimator
        elif nameof(estimator) == "bernoulli_nb":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: bnb_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["bernoulli_nb"] = estimator
        elif nameof(estimator) == "gaussian_nb":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: gnb_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["gaussian_nb"] = estimator
        elif nameof(estimator) == "linear_discriminant_analysis":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: lda_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["linear_discriminant_analysis"] = estimator
        elif nameof(estimator) == "quadratic_discriminant_analysis":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: qda_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["quadratic_discriminant_analysis"] = estimator
        elif nameof(estimator) == "xgb":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: xgb_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["xgb"] = estimator
        elif nameof(estimator) == "lgbm":
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: lgbm_objective(trial, x, y, estimator), n_trials=100)
            estimator.set_params(**study.best_params)
            estimator.fit(x, y)
            optimized_estimators["lgbm"] = estimator

    return optimized_estimators
