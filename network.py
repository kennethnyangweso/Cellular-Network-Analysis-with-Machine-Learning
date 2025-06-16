import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# =============================
# 🔧 TRAINING SECTION
# =============================
df = pd.read_csv("cellular_network_dataset")  # Make sure the path is correct

# Feature columns and target
features = [
    'latitude', 'longitude', 'signal_strength_(dbm)', 'latency_(ms)',
    'data_throughput_(mbps)', 'bb60c_measurement_(dbm)',
    'srsran_measurement_(dbm)', 'bladerfxa9_measurement_(dbm)',
    'hour', 'day_of_week', 'month'
]
target = 'network_type'

X = df[features]
y = df[target]

# Label encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# No categorical features in X, so passthrough
preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')

# Create pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, "best_network_type_model.pkl")
joblib.dump(label_encoder, "network_type_label_encoder.pkl")
print("✅ Model and encoder saved.")


# =============================
# 🌐 STREAMLIT APP SECTION
# =============================
st.title("📶 Network Type Predictor")

st.markdown("Enter the following information to predict **network type (3G, 4G, LTE, 5G)**:")

# User inputs
latitude = st.number_input("Latitude", value=0.0, format="%.6f")
longitude = st.number_input("Longitude", value=0.0, format="%.6f")
signal_strength = st.number_input("Signal Strength (dBm)", value=-85.0)
latency = st.number_input("Latency (ms)", value=1.0)
throughput = st.number_input("Data Throughput (Mbps)", value=1.0)

bb60c = st.number_input("BB60C Measurement (dBm)", value=-90.0)
srsran = st.number_input("SRSRAN Measurement (dBm)", value=-90.0)
bladerfx = st.number_input("BladeRFxA9 Measurement (dBm)", value=-90.0)

hour = st.slider("Hour of Day", 0, 23, 12)
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
month = st.selectbox("Month", list(range(1, 13)))

# Day mapping
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
           'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

# Predict button
if st.button("Predict Network Type"):
    input_data = pd.DataFrame([{
        'latitude': latitude,
        'longitude': longitude,
        'signal_strength_(dbm)': signal_strength,
        'latency_(ms)': latency,
        'data_throughput_(mbps)': throughput,
        'bb60c_measurement_(dbm)': bb60c,
        'srsran_measurement_(dbm)': srsran,
        'bladerfxa9_measurement_(dbm)': bladerfx,
        'hour': hour,
        'day_of_week': day_map[day_of_week],
        'month': month
    }])

    # Load model and encoder
    model = joblib.load("best_network_type_model.pkl")
    label_encoder = joblib.load("network_type_label_encoder.pkl")

    # Predict and decode
    prediction_encoded = model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    st.success(f"📡 Predicted Network Type: **{prediction_label}**")
