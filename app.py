import pandas as pd
import streamlit as st
from weather import get_current_temperature_sync, get_current_temperature_async
from data_processing import analyze_city_data
from visualization import visualize_temperature
import asyncio

API_KEY = "afb6147ee48eace31b567b026d07535e"
CITIES = ['Berlin', 'Moscow', 'Cairo', 'Dubai']

data = pd.read_csv("data/temperature_data.csv")


def compare_temperature(city, data):
    city_data = data[data['city'] == city]

    current_temp_sync = get_current_temperature_sync(city, API_KEY)
    st.write(f"Текущая температура в {city}: {current_temp_sync}°C")

    analysis = analyze_city_data(city_data)
    season_stats = analysis['season_stats']
    anomalies = analysis['anomalies']

    visualize_temperature(city_data, season_stats, anomalies)


st.sidebar.title("Мониторинг температуры")
selected_city = st.sidebar.selectbox("Выберите город", CITIES)

compare_temperature(selected_city, data)
