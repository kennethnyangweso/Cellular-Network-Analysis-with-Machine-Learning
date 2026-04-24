import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_random_forest_classifier.pkl")

st.title("📶 Network Type Classification")

st.write("Predict network type from signal and network conditions")

# ---------------- INPUTS ---------------- #

latitude = st.number_input(
    "Latitude",
    value=-1.286389
)

longitude = st.number_input(
    "Longitude",
    value=36.817223
)

# This is required by your classifier
signal_strength = st.number_input(
    "Signal Strength (dBm)",
    value=-85.0
)

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

hour = st.slider(
    "Hour",
    0,23,12
)

day_of_week = st.slider(
    "Day of Week",
    0,6,2
)

month = st.slider(
    "Month",
    1,12,6
)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning","Afternoon","Evening","Night"]
)

time_map = {
    "Morning":0,
    "Afternoon":1,
    "Evening":2,
    "Night":3
}

# ---------------- PREDICT ---------------- #

if st.button("Predict Network Type"):

    input_data = pd.DataFrame({
        'latitude':[latitude],
        'longitude':[longitude],
        'signal_strength_(dbm)':[signal_strength],
        'data_throughput_(mbps)':[throughput],
        'latency_(ms)':[latency],
        'hour':[hour],
        'day_of_week':[day_of_week],
        'month':[month],
        'time_of_day':[time_map[time_of_day]]
    })

    pred_encoded = model.predict(input_data)[0]

    # Decode prediction
    label_map = {
        0: "LTE",
        1: "3G",
        2: "4G",
        3: "5G"
    }

    prediction = label_map[pred_encoded]

    st.success(
        f"Predicted Network Type: {prediction}"
    )

    # Interpretation
    if prediction == "5G":
        st.success("High-speed next-generation network detected")

    elif prediction in ["4G", "LTE"]:
        st.info("Strong broadband cellular network")

    else:
        st.warning("Legacy network connection")

    