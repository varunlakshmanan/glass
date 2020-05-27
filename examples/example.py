from GlassRegressor import GlassRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

# Parse data into DataFrame
data_file_path = "D:/Documents/Projects/Glass/data.csv"
data = pd.read_csv(data_file_path)

# Select features and target column
features = ['age', 'male', 'female', 'symptom_onset_hospitalization', 'mortality_rate', 'pop_density',
            'high_risk_travel']
X = data[features]
y = data.death

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Use GlassRegressor
ensemble = GlassRegressor()
ensemble.fit(X_train, y_train, X_test, y_test)
ensemble.predict(X_test)
ensemble.describe()
