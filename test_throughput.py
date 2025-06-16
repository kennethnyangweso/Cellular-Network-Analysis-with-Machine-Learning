# train_throughput_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

# Load your dataset
df = pd.read_csv("cellular_network_dataset")  # Replace with your actual CSV file

# Feature & target selection
X = df.drop(columns=["throughput"])  # Replace with correct target column name
y = df["throughput"]


# Column categorization
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features)
])

# Pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "throughput_model.pkl")

print("Model saved successfully.")
