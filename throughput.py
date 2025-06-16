import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

# ================================
# 🚀 TRAINING SECTION
# ================================
df = pd.read_csv("cellular_network_dataset")  # Update to your correct file path

features = [
    'latitude', 'longitude', 'signal_strength_(dbm)', 'latency_(ms)', 'network_type',
    'bb60c_measurement_(dbm)', 'srsran_measurement_(dbm)', 'bladerfxa9_measurement_(dbm)',
    'hour', 'day_of_week', 'month'
]
target = 'data_throughput_(mbps)'

X = df[features]
y = df[target]

# One-hot encode 'network_type'
categorical_features = ['network_type']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42))
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save pipeline
joblib.dump(model, "best_throughput_xgb.pkl")
print("✅ Throughput model saved as best_throughput_xgb.pkl")


# ================================
# 🌐 STREAMLIT APP SECTION
# ================================
st.title("📡 Data Throughput Predictor")

st.markdown("Enter the following information to predict **data throughput (Mbps)**:")

# User inputs
latitude = st.number_input("Latitude", value=0.0, format="%.6f")
longitude = st.number_input("Longitude", value=0.0, format="%.6f")
signal_strength = st.number_input("Signal Strength (dBm)", value=-85.0)
latency = st.number_input("Latency (ms)", value=1.0)

network_type = st.selectbox("Network Type", ["3G", "4G", "LTE", "5G"])

bb60c = st.number_input("BB60C Measurement (dBm)", value=-90.0)
srsran = st.number_input("SRSRAN Measurement (dBm)", value=-90.0)
bladerfx = st.number_input("BladeRFxA9 Measurement (dBm)", value=-90.0)

hour = st.slider("Hour of Day", 0, 23, 12)
day_of_week = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
month = st.selectbox("Month", list(range(1, 13)))

# Encoding mappings
day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
           'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

# Predict
if st.button("Predict Throughput"):
    input_data = pd.DataFrame([{
        'latitude': latitude,
        'longitude': longitude,
        'signal_strength_(dbm)': signal_strength,
        'latency_(ms)': latency,
        'network_type': network_type,
        'bb60c_measurement_(dbm)': bb60c,
        'srsran_measurement_(dbm)': srsran,
        'bladerfxa9_measurement_(dbm)': bladerfx,
        'hour': hour,
        'day_of_week': day_map[day_of_week],
        'month': month
    }])

    # Load model
    model = joblib.load("best_throughput_xgb.pkl")

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"📶 Predicted Data Throughput: **{prediction:.2f} Mbps**")
