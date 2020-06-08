from GlassRegressor import GlassRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

data_file_path = "D:/Documents/Projects/Glass/data.csv"
data = pd.read_csv(data_file_path)

scaler = RobustScaler()

features = ['age', 'male', 'female', 'symptom_onset_hospitalization', 'mortality_rate', 'pop_density',
            'high_risk_travel']
X = data[features]
y = data.death
scaled_X = scaler.fit_transform(X)

# 70% training, 15% validation, 15% test
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.15)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=(0.15/0.85))

ensemble = GlassRegressor()
ensemble.fit(X_train, y_train, X_validation, y_validation, timeout=15, max_in_ensemble=4)
ensemble.describe(X_test, y_test)
