import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('artifacts/model.pkl') 
preprocessor = joblib.load('artifacts/preprocessor.pkl') 

st.title('Equipment Energy Consumption Prediction')


col1, col2 = st.columns(2)
with col1:
    outdoor_temp = st.number_input('Outdoor Temperature (°C)')
    zone_temp = st.number_input('Zone Temperature (°C)')
with col2:
    zone_humidity = st.number_input('Zone Humidity (%)', min_value=0.0, max_value=100.0)

hour = st.slider('Hour', 0, 23, 12)


hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
temp_diff = outdoor_temp - zone_temp


input_data = pd.DataFrame([[outdoor_temp, zone_temp, zone_humidity,
                           hour_sin, hour_cos, temp_diff]],
                         columns=['outdoor_temperature', 'zone_temperature',
                                  'zone_humidity', 'hour_sin',
                                  'hour_cos', 'temp_diff'])


if st.button('Predict Energy Consumption'):

    processed_data = preprocessor.transform(input_data)
    
 
    prediction = model.predict(processed_data)
    
    st.success(f'Predicted Equipment Energy Consumption: {prediction[0]:.2f} units')

