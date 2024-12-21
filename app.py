import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import requests
import aiohttp
import asyncio
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor

seasonal_temperatures = {
    "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
    "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
    "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
    "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
    "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
    "Sydney": {"winter": 12, "spring": 18, "summer": 25, "autumn": 20},
    "Berlin": {"winter": 0, "spring": 10, "summer": 20, "autumn": 11},
    "Beijing": {"winter": -2, "spring": 13, "summer": 27, "autumn": 16},
    "Rio de Janeiro": {"winter": 20, "spring": 25, "summer": 30, "autumn": 25},
    "Dubai": {"winter": 20, "spring": 30, "summer": 40, "autumn": 30},
    "Los Angeles": {"winter": 15, "spring": 18, "summer": 25, "autumn": 20},
    "Singapore": {"winter": 27, "spring": 28, "summer": 28, "autumn": 27},
    "Mumbai": {"winter": 25, "spring": 30, "summer": 35, "autumn": 30},
    "Cairo": {"winter": 15, "spring": 25, "summer": 35, "autumn": 25},
    "Mexico City": {"winter": 12, "spring": 18, "summer": 20, "autumn": 15},
}

month_to_season = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}


def generate_realistic_temperature_data(cities, num_years=10):
    dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
    data = []

    for city in cities:
        for date in dates:
            season = month_to_season[date.month]
            mean_temp = seasonal_temperatures[city][season]
            temperature = np.random.normal(loc=mean_temp, scale=5)
            data.append({"city": city, "timestamp": date, "temperature": temperature, "season": season})

    df = pd.DataFrame(data)
    return df


def get_current_temperature_sync(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    elif response.status_code == 401:
        st.error("Error 401: Unauthorized. Please check your API key.")
        return None
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None


async def get_current_temperature_async(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            else:
                st.error(f"Error fetching data: {response.status}")
                return None


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


def visualize_temperature(data, season_stats, anomalies, plot_type='line'):
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

    if plot_type == 'line':
        ax.plot(data['timestamp'], data['temperature'], label='Температура')
    elif plot_type == 'bar':
        ax.bar(data['timestamp'], data['temperature'], label='Температура')

    ax.scatter(anomalies['timestamp'], anomalies['temperature'], color='red', label='Аномалии')
    ax.set_xlabel("Дата")
    ax.set_ylabel("Температура (°C)")
    ax.legend()
    st.pyplot(fig)


def compare_temperature(cities, data, api_key, plot_type='line'):
    for city in cities:
        city_data = data[data['city'] == city]

        if not api_key:
            st.warning("API-ключ не введен. Данные о текущей температуре не будут отображены.")
            return

        current_temp_sync = get_current_temperature_sync(city, api_key)
        if current_temp_sync is not None:
            st.write(f"Текущая температура в {city}: {current_temp_sync}°C")

        analysis = analyze_city_data(city_data)
        season_stats = analysis['season_stats']
        anomalies = analysis['anomalies']

        visualize_temperature(city_data, season_stats, anomalies, plot_type)


def main():
    st.sidebar.title("Мониторинг температуры")

    uploaded_file = st.sidebar.file_uploader("Загрузите файл с данными", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Данные успешно загружены!")
    else:
        st.sidebar.warning("Пожалуйста, загрузите файл с данными.")

    cities = list(seasonal_temperatures.keys())

    selected_cities = st.sidebar.multiselect("Выберите города", cities, default=cities)

    api_key = st.sidebar.text_input("Введите API-ключ для OpenWeatherMap")

    plot_type = st.sidebar.selectbox("Выберите тип графика", ['line', 'bar'])

    if uploaded_file is not None and api_key:
        filtered_data = generate_realistic_temperature_data(selected_cities)

        compare_temperature(selected_cities, filtered_data, api_key, plot_type)

    st.sidebar.subheader("Дополнительные возможности")
    st.sidebar.write("1. Параллельный анализ всех выбранных городов")
    st.sidebar.write("2. Выбор различных типов графиков (линейный, столбчатый)")

    st.sidebar.button("Выполнить параллельный анализ")


if __name__ == "__main__":
    main()
