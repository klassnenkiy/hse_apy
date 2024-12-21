import matplotlib.pyplot as plt
import streamlit as st


def visualize_temperature(data, season_stats, anomalies):
    st.title("Мониторинг температуры")

    mean_temp = season_stats['temperature']['mean'].mean()
    min_temp = season_stats['temperature']['mean'].min()
    max_temp = season_stats['temperature']['mean'].max()

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
