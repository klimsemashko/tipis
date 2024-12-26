import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Загрузка модели и скейлера
model = joblib.load('house_price_model.pkl')
scaler = joblib.load('scaler.pkl')

# Функция для получения ввода пользователя
def get_user_input():
    house_age = st.number_input("Возраст дома (лет)", min_value=0, max_value=100, value=10)
    distance_to_mrt = st.number_input("Расстояние до ближайшей станции MRT (в милях)", min_value=0.0, value=1.0)
    num_convenience_stores = st.number_input("Количество магазинов поблизости", min_value=0, value=1)
    latitude = st.number_input("Широта дома", value=25.033)
    longitude = st.number_input("Долгота дома", value=121.565)

    user_data = {
        'X2 house age': house_age,
        'X3 distance to the nearest MRT station': distance_to_mrt,
        'X4 number of convenience stores': num_convenience_stores,
        'X5 latitude': latitude,
        'X6 longitude': longitude
    }

    # Преобразуем данные в DataFrame
    input_df = pd.DataFrame([user_data])

    return input_df

# Основной блок Streamlit
st.title("Прогнозирование цены дома")

# Получаем данные от пользователя
user_input = get_user_input()

# Показываем введенные данные
st.write("Введенные данные:")
st.write(user_input)

# Кнопка для предсказания
if st.button("Прогнозировать цену"):
    # Масштабируем данные с использованием сохранённого скейлера
    user_input_scaled = scaler.transform(user_input)

    # Прогнозируем цену
    predicted_price = model.predict(user_input_scaled)[0]

    # Выводим результат
    st.subheader(f"Предсказанная цена за единицу площади(фут): {predicted_price:.2f}")

