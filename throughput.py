# throughput.py

import streamlit as st
import pandas as pd
import joblib

# Load preprocessor and model
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("throughput_model.pkl")

st.title("Throughput Prediction")

# Input form
latitude = st.number_input("Latitude", value=37.7749)
longitude = st.number_input("Longitude", value=-122.4194)
signal_strength = st.number_input("Signal Strength (dBm)", value=-85)
latency = st.number_input("Latency (ms)", value=30)
network_type = st.selectbox("Network Type", ["4G", "5G", "3G"])
bb60c = st.number_input("BB60C Measurement (dBm)", value=-90)
srsran = st.number_input("SRSRAN Measurement (dBm)", value=-88)
blade = st.number_input("BladeRFxA9 Measurement (dBm)", value=-87)
hour = st.slider("Hour", 0, 23, 12)
day_of_week = st.selectbox("Day of Week", list(range(7)))  # 0=Monday
month = st.selectbox("Month", list(range(1, 13)))
time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])

if st.button("Predict Throughput"):
    input_df = pd.DataFrame([{
        'latitude': latitude,
        'longitude': longitude,
        'signal_strength_(dbm)': signal_strength,
        'latency_(ms)': latency,
        'network_type': network_type,
        'bb60c_measurement_(dbm)': bb60c,
        'srsran_measurement_(dbm)': srsran,
        'bladerfxa9_measurement_(dbm)': blade,
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'time_of_day': time_of_day
    }])
    
    prediction = model.predict(input_df)
    st.success(f"Predicted Throughput: {prediction[0]:.2f} Mbps")

