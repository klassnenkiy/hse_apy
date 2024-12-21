import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import requests
import aiohttp
import asyncio
from sklearn.linear_model import LinearRegression


API_KEY = "afb6147ee48eace31b567b026d07535e"
CITIES = ['Berlin', 'Moscow', 'Cairo', 'Dubai']


data = pd.read_csv("data/temperature_data.csv")


def get_current_temperature_sync(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    elif response.status_code == 401:
        raise Exception("Error 401: Unauthorized. Please check your API key.")
    else:
        raise Exception(f"Error fetching data: {response.status_code}")

async def get_current_temperature_async(city, api_key=API_KEY):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            else:
                raise Exception(f"Error fetching data: {response.status}")

def clean_and_convert_data(city_data):
    city_data['temperature'] = pd.to_numeric(city_data['temperature'], errors='coerce')
    city_data['timestamp'] = pd.to_datetime(city_data['timestamp'], errors='coerce')
    city_data['rolling_mean'] = city_data['temperature'].rolling(window=30, center=True).mean()
    city_data['rolling_std'] = city_data['temperature'].rolling(window=30, center=True).std()
    city_data.dropna(subset=['temperature', 'rolling_mean', 'rolling_std'], inplace=True)

    return city_data

def analyze_city_data(city_data):
    city_data = clean_and_convert_data(city_data)
    city_data['anomaly'] = ((city_data['temperature'] > city_data['rolling_mean'] + 2 * city_data['rolling_std']) |
                            (city_data['temperature'] < city_data['rolling_mean'] - 2 * city_data['rolling_std']))
    season_stats = city_data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    city_data['timestamp_numeric'] = (city_data['timestamp'] - city_data['timestamp'].min()).dt.total_seconds()
    X = city_data['timestamp_numeric'].values.reshape(-1, 1)
    y = city_data['temperature'].values
    model = LinearRegression()
    model.fit(X, y)
    trend_slope = model.coef_[0]

    return {
        'season_stats': season_stats,
        'trend_slope': trend_slope,
        'anomalies': city_data[city_data['anomaly'] == True]
    }

def visualize_temperature(data, season_stats, anomalies):
    st.title("Мониторинг температуры")
    mean_temp = season_stats['mean'].mean()
    min_temp = season_stats['mean'].min()
    max_temp = season_stats['mean'].max()

    st.write(f"Средняя температура: {mean_temp}°C")
    st.write(f"Минимальная температура: {min_temp}°C")
    st.write(f"Максимальная температура: {max_temp}°C")
    st.subheader("Сезонный профиль")
    st.write(season_stats)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['timestamp'], data['temperature'], label='Температура')
    ax.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', label='Аномалии')
    ax.set_xlabel("Дата")
    ax.set_ylabel("Температура (°C)")
    ax.legend()
    st.pyplot(fig)

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
