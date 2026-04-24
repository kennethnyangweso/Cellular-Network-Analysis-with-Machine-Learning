import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_random_forest_regressor.pkl")

st.title("📡 Signal Strength Prediction")

# Inputs
latitude = st.number_input("Latitude", value=-1.286389)
longitude = st.number_input("Longitude", value=36.817223)

throughput = st.number_input(
    "Data Throughput (Mbps)",
    min_value=0.0,
    value=50.0
)

latency = st.number_input(
    "Latency (ms)",
    min_value=0.0,
    value=30.0
)

# Missing features added
srsran = st.number_input(
    "SRSRAN Measurement (dBm)",
    value=-85.0
)

bladerfx = st.number_input(
    "BladeRFxA9 Measurement (dBm)",
    value=-87.0
)

network_type = st.selectbox(
    "Network Type",
    ["3G","4G","LTE","5G"]
)

hour = st.slider("Hour",0,23,12)
day_of_week = st.slider("Day of Week",0,6,2)
month = st.slider("Month",1,12,6)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning","Afternoon","Evening","Night"]
)

# Match training encoding
network_map = {
    "3G":0,
    "4G":1,
    "LTE":2,
    "5G":3
}

time_map = {
    "Morning":0,
    "Afternoon":1,
    "Evening":2,
    "Night":3
}

if st.button("Predict Signal Strength"):

    # IMPORTANT:
    # Keep exact same order as training
    input_data = pd.DataFrame({
        'latitude':[latitude],
        'longitude':[longitude],
        'data_throughput_(mbps)':[throughput],
        'latency_(ms)':[latency],
        'network_type':[network_map[network_type]],
        'srsran_measurement_(dbm)':[srsran],
        'bladerfxa9_measurement_(dbm)':[bladerfx],
        'hour':[hour],
        'day_of_week':[day_of_week],
        'month':[month],
        'time_of_day':[time_map[time_of_day]]
    })

    prediction = model.predict(input_data)[0]

    st.success(
        f"Predicted Signal Strength: {prediction:.2f} dBm"
    )

    # Signal quality interpretation
    if prediction >= -70:
        st.success("Excellent Signal")
    elif prediction >= -90:
        st.warning("Moderate Signal")
    else:
        st.error("Weak Signal")