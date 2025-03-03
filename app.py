import streamlit as st
import numpy as np
import pandas as pd
import pickle
import math
import datetime

# Load trained model, PCA, and StandardScaler
with open("model_delivery.pkl", "rb") as f:
    model = pickle.load(f)

with open("pca_delivery.pkl", "rb") as f:
    pca = pickle.load(f)

with open("scaler_delivery.pkl", "rb") as f:
    scaler = pickle.load(f)

# Features to scale
features_to_scale = ['delivery_distance', 'log_delivery_distance', 'accept_hour_of_day',
                     'accept_day_of_week', 'delivery_hour_of_day', 'delivery_day_of_week']

# Streamlit App
st.title("Delivery ETA Prediction")

# User Inputs
region_id = st.number_input("Region ID", value=1)
lng = st.number_input("Longitude", value=0.0)
lat = st.number_input("Latitude", value=0.0)
aoi_id = st.number_input("AOI ID", value=1)
aoi_type = st.number_input("AOI Type", value=1)
accept_gps_lng = st.number_input("Accept GPS Longitude", value=0.0)
accept_gps_lat = st.number_input("Accept GPS Latitude", value=0.0)
delivery_gps_lng = st.number_input("Delivery GPS Longitude", value=0.0)
delivery_gps_lat = st.number_input("Delivery GPS Latitude", value=0.0)

# Accept & Delivery Times
accept_time = st.time_input("Accept Time", datetime.time(12, 0))
delivery_time = st.time_input("Delivery Time", datetime.time(12, 30))

# Feature Engineering
accept_dt = datetime.datetime.combine(datetime.date.today(), accept_time)
delivery_dt = datetime.datetime.combine(datetime.date.today(), delivery_time)

# Compute Distance and Log Distance
delivery_distance = math.sqrt((delivery_gps_lng - accept_gps_lng) ** 2 + (delivery_gps_lat - accept_gps_lat) ** 2)
log_delivery_distance = np.log1p(delivery_distance)

# Extract Date Features
accept_hour_of_day = accept_dt.hour
accept_day_of_week = accept_dt.weekday()
delivery_hour_of_day = delivery_dt.hour
delivery_day_of_week = delivery_dt.weekday()

# Create DataFrame
input_data = pd.DataFrame({
    "region_id": [region_id],
    "lng": [lng],
    "lat": [lat],
    "aoi_id": [aoi_id],
    "aoi_type": [aoi_type],
    "accept_gps_lng": [accept_gps_lng],
    "accept_gps_lat": [accept_gps_lat],
    "delivery_gps_lng": [delivery_gps_lng],
    "delivery_gps_lat": [delivery_gps_lat],
    "accept_hour_of_day": [accept_hour_of_day],
    "accept_day_of_week": [accept_day_of_week],
    "delivery_hour_of_day": [delivery_hour_of_day],
    "delivery_day_of_week": [delivery_day_of_week],
    "delivery_distance": [delivery_distance],
    "log_delivery_distance": [log_delivery_distance],
})

# Scale selected features
input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])

# Apply PCA
pca_transformed = pca.transform(input_data)

# Combine original & PCA-transformed features
final_input = np.hstack((input_data, pca_transformed))

# Predict Button
if st.button("Predict ETA"):
    prediction = model.predict(final_input)
    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")
