## What is Glass?
Glass is the **Generalizable**, **Loss-minimizing**, **Automated** **Sklearn** **System**. Glass is a template for quickly building and deploying a high-performance model ensemble on tabular and numerical data. It returns the best possible fitted model ensemble out of every possible combination of estimators whose parameters have been hyperoptimized. Ordinarily, machine learning requires five core steps:
 * Data Preprocessing
 * Feature Engineering 
 * Model Fitting
 * Hyperparameter Optimization
 * Model Evaluation and Ensembling 
 
Using Glass greatly minimizes the steps of model fitting, hyperparameter optimization, and model ensembling. Because it's built on top of native scikit-learn functions, it's immediately familiar to understand. You can deploy a high-performance model in under 5 lines of code. 

## How do I use Glass?
Preprocess your dataset, and perform initial feature engineering. When preprocessing, impute missing data and encode categorical features as necessary. Ensure that your data has no null or NaN values. When feature engineering, create custom features and drop unnecessary columns based on feature importance. 

After that, follow the instructions in the [Documentation](https://github.com/varunlakshmanan/glass/wiki/Documentation) to get started. Take a look at the [example provided](https://github.com/varunlakshmanan/glass/blob/master/examples/example.py) for further usage guidelines.  

## How does Glass work? 
Glass works by initializing many different scikit-learn models with varying architectures. It fits each of them to your input data while training, and hyperoptimizes the parameters of each model. The accuracy of every possible ensembled combination of these models is then ranked, and the best fitted model ensemble is returned for your use. 

## What technologies does Glass use? 
Languages:
* Python

Frameworks: 
* Scikit-learn - for most of the estimators, predictors, and transformers that compose the final model
* Pandas - for data manipulation 
* XGBoost 
* LightGBM 
* Optuna - for hyperparameter optimization 
