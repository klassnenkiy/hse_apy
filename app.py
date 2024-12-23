import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import aiohttp
import time
import requests
from sklearn.linear_model import LinearRegression
from io import BytesIO
from aiocache import cached
from aiocache.serializers import JsonSerializer
from concurrent.futures import ThreadPoolExecutor, as_completed

seasonal_temperatures = {
    "Moscow": {"winter": -10, "spring": 5, "summer": 18, "autumn": 8},
    "New York": {"winter": 0, "spring": 10, "summer": 25, "autumn": 15},
    "London": {"winter": 5, "spring": 11, "summer": 18, "autumn": 12},
    "Paris": {"winter": 4, "spring": 12, "summer": 20, "autumn": 13},
    "Tokyo": {"winter": 6, "spring": 15, "summer": 27, "autumn": 18},
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


def generate_realistic_temperature_data(city, num_years=10):
    dates = pd.date_range(start="2010-01-01", periods=365 * num_years, freq="D")
    data = []
    for date in dates:
        season = month_to_season[date.month]
        mean_temp = seasonal_temperatures[city][season]
        temperature = np.random.normal(loc=mean_temp, scale=5)
        data.append({"city": city, "timestamp": date, "temperature": temperature, "season": season})
    df = pd.DataFrame(data)
    return df


@cached(ttl=60, serializer=JsonSerializer())
async def get_current_temperature_async(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data['main']['temp']
            elif response.status == 401:
                st.error("Ошибка: Неверный API-ключ. Пожалуйста, проверьте ваш ключ.")
                return None
            else:
                st.error(f"Ошибка при получении данных для {city}: {response.status}")
                return None


def clean_and_convert_data(city_data):
    city_data['temperature'] = pd.to_numeric(city_data['temperature'], errors='coerce')
    city_data['timestamp'] = pd.to_datetime(city_data['timestamp'], errors='coerce')
    city_data['rolling_mean'] = city_data['temperature'].rolling(window=30, center=True).mean()
    city_data['rolling_std'] = city_data['temperature'].rolling(window=30, center=True).std()
    city_data.dropna(subset=['temperature', 'rolling_mean', 'rolling_std'], inplace=True)
    return city_data


def analyze_city_data(city_data, sensitivity=2.0):
    city_data = clean_and_convert_data(city_data)
    city_data['anomaly'] = (
            (city_data['temperature'] > city_data['rolling_mean'] + sensitivity * city_data['rolling_std']) |
            (city_data['temperature'] < city_data['rolling_mean'] - sensitivity * city_data['rolling_std']))
    season_stats = city_data.groupby('season')['temperature'].agg(['mean', 'std', 'min', 'max']).reset_index()
    city_data['year'] = city_data['timestamp'].dt.year
    trend_per_year = []
    for year, group in city_data.groupby('year'):
        group['timestamp_numeric'] = (group['timestamp'] - group['timestamp'].min()).dt.total_seconds()
        X = group['timestamp_numeric'].values.reshape(-1, 1)
        y = group['temperature'].values
        model = LinearRegression()
        model.fit(X, y)

        trend_per_year.append({
            'year': year,
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'trend_direction': "положительный" if model.coef_[0] > 0 else "отрицательный" if model.coef_[0] < 0 else "плоский"
        })
    trend_per_season = []
    for season, group in city_data.groupby('season'):
        group['timestamp_numeric'] = (group['timestamp'] - group['timestamp'].min()).dt.total_seconds()
        X = group['timestamp_numeric'].values.reshape(-1, 1)
        y = group['temperature'].values
        model = LinearRegression()
        model.fit(X, y)

        trend_per_season.append({
            'season': season,
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'trend_direction': "положительный" if model.coef_[0] > 0 else "отрицательный" if model.coef_[0] < 0 else "плоский"
        })

    trend_per_year = pd.DataFrame(trend_per_year).sort_values(by='year')
    trend_per_season = pd.DataFrame(trend_per_season).sort_values(by='season')
    city_data['timestamp_numeric'] = (city_data['timestamp'] - city_data['timestamp'].min()).dt.total_seconds()
    X = city_data['timestamp_numeric'].values.reshape(-1, 1)
    y = city_data['temperature'].values
    model = LinearRegression()
    if len(X) > 1:
        model.fit(X, y)
        overall_trend_slope = model.coef_[0]
        overall_trend_direction = "положительный" if overall_trend_slope > 0 else "отрицательный" if overall_trend_slope < 0 else "плоский"
    else:
        overall_trend_slope = None
        overall_trend_direction = "неопределено"

    return {
        'season_stats': season_stats,
        'trend_per_year': trend_per_year,
        'trend_per_season': trend_per_season,
        'overall_trend_slope': overall_trend_slope,
        'overall_trend_direction': overall_trend_direction,
        'anomalies': city_data[city_data['anomaly'] == True]
    }



def get_temperature_color(temp):
    if temp > 30:
        return 'red'
    elif temp < 0:
        return 'blue'
    else:
        return 'green'


def get_temperature_emoji(temp):
    if temp > 30:
        return "🥵"
    elif temp < 0:
        return "🥶"
    else:
        return "☀️"


def display_temperature_data(temperatures):
    st.subheader("Текущие температуры по городам")
    df = pd.DataFrame(list(temperatures.items()), columns=["Город", "Температура"])
    df['Эмодзи'] = df['Температура'].apply(lambda x: get_temperature_emoji(x))
    df['Цвет'] = df['Температура'].apply(lambda x: get_temperature_color(x))

    st.write("**Текущие температуры** по выбранным городам:")
    st.table(df[['Город', 'Температура', 'Эмодзи']])


def visualize_temperature_by_year(city, plot_data, selected_years):
    city_data = plot_data[plot_data['city'] == city].copy()
    city_data['year'] = city_data['timestamp'].dt.year
    city_data['day_of_year'] = city_data['timestamp'].dt.dayofyear
    city_data_filtered = city_data[city_data['year'].isin(selected_years)]

    fig = px.line(city_data_filtered, x='day_of_year', y='temperature', color='year',
                  title=f'Температура в {city} ({", ".join(map(str, selected_years))})')
    fig.update_layout(
        xaxis_title='Day of Year',
        yaxis_title='Temperature (°C)',
        template='plotly_white'
    )

    for year in selected_years:
        year_data = city_data_filtered[city_data_filtered['year'] == year]
        year_data['timestamp_numeric'] = (year_data['timestamp'] - year_data['timestamp'].min()).dt.total_seconds()

        X = year_data['timestamp_numeric'].values.reshape(-1, 1)
        y = year_data['temperature'].values
        model = LinearRegression()
        model.fit(X, y)

        trend_slope = model.coef_[0]
        trend_intercept = model.intercept_

        year_data['trend'] = model.predict(X)

        fig.add_scatter(
            x=year_data['day_of_year'],
            y=year_data['trend'],
            mode='lines',
            line=dict(color='black', dash='dash'),
            name=f"Тренд {year}"
        )

    st.plotly_chart(fig)



def visualize_temperature(data, season_stats, anomalies, plot_type='line', city=None, trend_direction=None,
                          trend_slope=None, trend_per_season=None):
    mean_temp = season_stats['mean'].mean()
    min_temp = season_stats['min'].min()
    max_temp = season_stats['max'].max()

    st.write(f"Средняя температура: **{mean_temp:.2f}°C**")
    st.write(f"Минимальная температура: **{min_temp:.2f}°C**")
    st.write(f"Максимальная температура: **{max_temp:.2f}°C**")

    if trend_direction is not None and trend_slope is not None:
        st.write(f"Общий тренд: _{trend_direction}_")

    st.subheader(f"Сезонный профиль для города {city}")
    st.write(season_stats)
    if not anomalies.empty:
        st.markdown("### 🚨 **Аномалии температуры** 🚨")
        st.write(anomalies)
    else:
        st.markdown("### 🔍 **Нет аномалий температуры** 🔍")

    fig = None
    if plot_type == 'line':
        fig = px.line(data, x='timestamp', y='temperature', title=f"Температура в {city}")
        fig.add_scatter(x=anomalies['timestamp'], y=anomalies['temperature'], mode='markers', marker=dict(color='red'),
                        name="Аномалии")
    elif plot_type == 'bar':
        fig = px.bar(data, x='timestamp', y='temperature', title=f"Температура в {city}", color='temperature',
                     color_continuous_scale='Viridis', opacity=0.8)
        fig.update_traces(marker=dict(line=dict(width=0)))
        fig.add_scatter(x=anomalies['timestamp'], y=anomalies['temperature'], mode='markers',
                        marker=dict(color='red', size=8), name="Аномалии")

    plot_key = f"{city}_{plot_type}_temperature_plot_{int(time.time())}"
    st.plotly_chart(fig, key=plot_key)

    if trend_per_season is not None:
        st.subheader("Тренды по сезонам")
        if isinstance(trend_per_season, pd.DataFrame):
            for idx, row in trend_per_season.iterrows():
                st.write(f"Сезон _{row['season']}_ : Тренд **{row['trend_direction']}**")
        else:
            st.warning("Нет данных о тренде по сезонам.")

    st.subheader("Тренды по годам")
    trend_per_year = analyze_city_data(data)['trend_per_year']
    for index, row in trend_per_year.iterrows():
        st.write(f"Год **{row['year']}**: Тренд _{row['trend_direction']}_")


def generate_excel_report(data):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, sheet_name="Temperature Data")
    return output.getvalue()


def get_temperatures_for_multiple_cities_parallel(cities, api_key):
    temperatures = {}
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_current_temperature_sync, city, api_key): city for city in cities}
        for future in as_completed(futures):
            city = futures[future]
            try:
                temperature = future.result()
                if temperature is not None:
                    temperatures[city] = temperature
            except Exception as e:
                st.error(f"Ошибка при получении данных для {city}: {e}")
    return temperatures


def get_current_temperature_sync(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['main']['temp']
    elif response.status_code == 401:
        st.error("Ошибка: Неверный API-ключ. Пожалуйста, проверьте ваш ключ.")
        return None
    else:
        st.error(f"Ошибка при получении данных для {city}: {response.status_code}")
        return None

def main():
    st.sidebar.title("Мониторинг температуры")
    uploaded_file = st.sidebar.file_uploader("Загрузите файл с данными", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Данные успешно загружены!")
        selected_years = st.sidebar.multiselect("Выберите года для отображения",
                                                pd.to_datetime(data['timestamp']).dt.year.unique())
    else:
        data = None
        selected_years = None
    cities = list(seasonal_temperatures.keys())
    selected_city = st.sidebar.selectbox("Выберите город", cities)
    api_key = st.sidebar.text_input("Введите API-ключ для OpenWeatherMap")
    plot_type = st.sidebar.radio("Выберите тип графика", ('line', 'bar'))
    sensitivity = st.sidebar.slider("Чувствительность для аномалий (множитель стандартного отклонения)", 1.0, 3.0, 2.0)
    filtered_data = generate_realistic_temperature_data(selected_city)
    method = st.sidebar.radio("Выберите метод получения температуры", ("Синхронный", "Параллельный"))
    if method == "Параллельный":
        if uploaded_file is not None and api_key:
            cities_selected = st.sidebar.multiselect("Выберите города для параллельных запросов", cities)
            if cities_selected:
                start_time = time.time()
                temperatures = get_temperatures_for_multiple_cities_parallel(cities_selected, api_key)
                end_time = time.time()
                st.write(f"Время выполнения параллельных запросов: {end_time - start_time:.2f} секунд")
                display_temperature_data(temperatures)
                st.write(temperatures)
                all_data = []
                for city in cities_selected:
                    city_data = generate_realistic_temperature_data(city)
                    all_data.append(city_data)
                combined_data = pd.concat(all_data)
                season_stats = analyze_city_data(combined_data, sensitivity)['season_stats']
                anomalies = analyze_city_data(combined_data, sensitivity)['anomalies']
                trend_slope = analyze_city_data(combined_data, sensitivity)['overall_trend_slope']
                trend_direction = analyze_city_data(combined_data, sensitivity)['overall_trend_direction']
                analysis = analyze_city_data(filtered_data, sensitivity)
                trend_per_season = analysis['trend_per_season']
                for city in cities_selected:
                    st.subheader(f"Температура в городе {city}")
                    city_data = combined_data[combined_data['city'] == city]
                    city_season_stats = season_stats[season_stats['season'].isin(city_data['season'].unique())]
                    city_anomalies = anomalies[anomalies['city'] == city]
                    visualize_temperature(city_data, city_season_stats, city_anomalies, plot_type, city,
                                          trend_direction, trend_slope, trend_per_season)
                    if selected_years:
                        visualize_temperature_by_year(city, combined_data, selected_years)

    if uploaded_file is not None and api_key:
        if method == "Синхронный":
            start_time = time.time()
            current_temp = get_current_temperature_sync(selected_city, api_key)
            end_time = time.time()
            st.write(f"Время выполнения синхронного запроса: {end_time - start_time:.2f} секунд")
            if current_temp is not None:
                st.subheader(f"Температура в городе {selected_city}")
                st.write(f"Текущая температура в {selected_city}: {current_temp}°C")
                current_season = month_to_season[pd.to_datetime('today').month]
                normal_temp = seasonal_temperatures[selected_city][current_season]
                season_data = filtered_data[filtered_data['season'] == current_season]
                std_dev = season_data['temperature'].std()
                if abs(current_temp - normal_temp) > 2 * std_dev:
                    st.warning(
                        f"Текущая температура в {selected_city} отклоняется от нормы для сезона {current_season}.")
                else:
                    st.success(
                        f"Текущая температура в {selected_city} соответствует нормам для сезона {current_season}.")
            analysis = analyze_city_data(filtered_data, sensitivity)
            season_stats = analysis['season_stats']
            anomalies = analysis['anomalies']
            trend_direction = analysis.get('overall_trend_direction', 'неопределено')
            trend_slope = analysis.get('overall_trend_slope', None)
            trend_per_season = analysis['trend_per_season']
            visualize_temperature(filtered_data, season_stats, anomalies, plot_type, selected_city, trend_direction,
                                  trend_slope, trend_per_season)
            if selected_years:
                visualize_temperature_by_year(selected_city, filtered_data, selected_years)

            with st.expander(f"Исторические данные для города {selected_city}", expanded=True):
                st.dataframe(filtered_data)

        st.sidebar.subheader("Дополнительные возможности")
        st.sidebar.write("1. Выбор различных типов графиков (линейный, столбчатый)")
        st.sidebar.write("2. Настройка чувствительности для выявления аномалий")
        st.sidebar.write("3. Сравнение температур по нескольким городам")
        st.sidebar.write("4. Возможность скачивания отчета в Excel")
        st.sidebar.write("5. Кэширование")
        st.sidebar.download_button("Скачать отчет (Excel)", generate_excel_report(data),
                                   file_name="temperature_report.xlsx")


if __name__ == "__main__":
    main()
