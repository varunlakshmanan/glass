## What is Glass?
Glass is the **Generalizable**, **Loss-minimizing**, **Automated** **Sklearn** **System**. Glass is a template for quickly building and deploying a high-performance model ensemble on non-image, non-NLP data. Ordinarily, machine learning requires five core steps:
 * Data Preprocessing
 * Model Fitting
 * Hyperparameter Optimization
 * Feature Engineering 
 * Model Evaluation and Ensembling 
 
Using Glass as a starter template greatly minimizes the steps of model fitting, hyperparameter optimization, and model ensembling. Because it's built on top of native scikit-learn functions, it's immediately familiar to understand.

## How does Glass work? 
...

## How do I use Glass?
Preprocess your dataset, and perform initial feature engineering. When preprocessing, impute missing data and encode categorical features as necessary. Ensure that your data has no null or NaN values. When feature engineering, create custom features and drop unnecessary columns based on feature importance. 

Call the `predict()` function and pass in your preprocesssed `DataFrame` as well as the indicator string, either "regressor" or "classifier". (Use a regressor for numerical/continuous output and a classifier for categorical output.) `predict()` returns a `VotingClassifier` object that has been fitted. 

The specific parameters used in each step of the `predict()` function can be manually modified if needed prior to calling the function. (Eventually, this functionality will be accepted as optional parameters for `predict()`.) 

## What technologies does Glass use? 
Languages:
* Python

Frameworks: 
* Scikit-learn - for most of the estimators, predictors, and transformers that compose the final model
* Pandas - for data manipulation 
* XGBoost 
* LightGBM 
* Optuna - for hyperparameter optimization 
