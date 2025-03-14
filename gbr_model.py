# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 00:27:59 2025

@author: Aftab
"""

import pickle
import streamlit as st
import pandas as pd

# opening model in read binary mode
load = open('gbr.pkl','rb')
model = pickle.load(load)

st.set_page_config(page_title="Solar Power Prediction", layout="wide")    
page_bg_img = '''
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1508514177221-188b1cf16e9d?q=80&w=2072&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
        background-size: cover;
        background-attachment: fixed;
        background-position:top right;
    }
    
    [data-testid="stAppViewContainer"], h1, h2, h3, h4, h5, h6, p {
        color: white !important;
    }
    
    #power_output,power_output b{
        color:black !important;
        font-size:28px;
        font-weight:bold;
    }
    
    div[data-baseweb="input"] label {
        color: white !important;
    }

    .stButton > button {
            background-color: black !important;
            color: white !important;
            font-weight: bold !important;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
    }

    .stButton > button:hover {
            background-color: gray !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
       background-color: white !important;
   }

   .stSelectbox div[data-baseweb="single-value"] > div {
       color: black !important;
   }

   /* Dropdown items background & text color */
   .stSelectbox div[data-baseweb="option"] >div {
       background-color: white !important;
       color: black !important;
   }
</style>
'''

# Injecting CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


# Defining Prediction Function

def predict(distance_to_solar_noon,temperature,wind_direction,wind_speed,
            sky_cover,visibility,humidity,avg_wind_speed_period,avg_pressure_period):
    prediction = model.predict([[distance_to_solar_noon,temperature,wind_direction,wind_speed,
                sky_cover,visibility,humidity,avg_wind_speed_period,avg_pressure_period]])
    return prediction


if 'data' not in st.session_state:
    st.session_state.data = []


def main():
    
    st.title('☀️ Solar Power Generation Predictor')

    st.markdown("### Enter Weather and Solar Parameters:")
        
    col1,col2,col3 = st.columns(3)
    
    with col1:
        distance_to_solar_noon = st.number_input('Distance to solar noon (in radians)',step=0.000001,format="%.6f",min_value=0.0)
        temperature = st.number_input('Daily average temperature (in degree celsius)',step=0.01,format="%.2f",max_value=50.0)
        wind_direction = st.number_input('Daily average wind Direction (in degrees)',step=0.01,format="%.2f",min_value=0.0,max_value=360.0)
    
    with col2:
        wind_speed = st.number_input('Daily average wind speed (in m/s)',format="%.2f",step=0.01)
        sky_cover_options = [
            "0 - Clear Sky ☀️",
            "1 - Few Clouds 🌤",
            "2 - Partly Cloudy ⛅",
            "3 - Mostly Cloudy ☁️",
            "4 - Overcast 🌫"
        ]
        sky_cover_desc = st.selectbox(label='Sky cover',options=sky_cover_options)
        sky_cover = sky_cover_options.index(sky_cover_desc)  # Get numeric value
        visibility = st.number_input('Visibility (in kms)',format="%.2f",step=0.01)
        
    with col3:
        humidity = st.number_input('Humidity (in %)',format="%.2f",step=0.01,min_value=0.0,max_value=100.0)
        avg_wind_speed_period = st.number_input('Average Wind Speed during 3 hours (in m/s)',format="%.2f",step=0.01)
        avg_pressure_period = st.number_input('Average barometric pressure during 3 hours (in mercury/inch)',format="%.2f",step=0.01)
    
    
    if st.button('Predict'):
        result = predict(distance_to_solar_noon, temperature, wind_direction, wind_speed, sky_cover, visibility, humidity, avg_wind_speed_period, avg_pressure_period)
        
        st.markdown(f"<span id='power_output'>Approximately {result[0]:.2f} Jules of power will be generated in the interval of 3 hours.</span>",unsafe_allow_html=True)
        
        st.session_state.data.append({
            'Distance to Solar Noon': distance_to_solar_noon,
            'Temperature': temperature,
            'Wind Direction': wind_direction,
            'Wind Speed': wind_speed,
            'Sky Cover': sky_cover,
            'Visibility': visibility,
            'Humidity': humidity,
            'Avg Wind Speed (Period)': avg_wind_speed_period,
            'Avg Pressure (Period)': avg_pressure_period,
            'Predicted Power': result
            })
        
    if st.session_state.data:
        df = pd.DataFrame(st.session_state.data)
        st.write("### Previous Predictions")
        st.dataframe(df)
        

        
if __name__ == '__main__':
    main()
