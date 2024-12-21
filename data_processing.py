import pandas as pd
from sklearn.linear_model import LinearRegression


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