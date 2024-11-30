import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

st.title("Prediksi Harga Bitcoin Dengan Regresi Polynomial")

st.sidebar.title("Navigation")
with st.sidebar :
    page = option_menu ("Pilih Halaman", ["Home", "Data Understanding","Preprocessing", "Model", "Evaluasi","Testing"], default_index=0)

if app_mode == "Eksplorasi Data":
    st.header("1. Eksplorasi Data")

    url = "https://raw.githubusercontent.com/Wafxd/PSD/refs/heads/main/bitcoin_price_Training%20-%20Training.csv"
    data = pd.read_csv(url)

    st.write("Data Mentah:")
    st.write(data)

    data['Volume'] = pd.to_numeric(data['Volume'].str.replace(',', ''), errors='coerce')
    data['Market Cap'] = pd.to_numeric(data['Market Cap'].str.replace(',', ''), errors='coerce')
    data['Date'] = pd.to_datetime(data['Date'])

    st.write("Pengecekan Missing Values")
    st.write(data.isnull().sum())

    data.fillna(data.mean(), inplace=True)

    st.write("Missing Values diisi dengan Rata-Rata")
    st.write(data.isnull().sum())

    st.write("Dataset Describe:")
    st.write(data.describe())

    st.subheader("Harga Penutupan Bitcoin")
    fig, ax = plt.subplots(figsize=(14, 7))  
    ax.plot(data['Date'], data['Close'], label='Close Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title('Harga Penutupan Bitcoin')
    ax.legend()

    st.pyplot(fig)

    numeric_data = data.select_dtypes(include=[np.number])
    
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))  
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)  
    ax.set_title('Correlation Heatmap')

    st.pyplot(fig)

elif app_mode == "Prediksi":
    st.header("2. Prediksi")

    with open('polynomial_regression_model.pkl', 'rb') as file:
        loaded_model, loaded_poly, loaded_scaler = pickle.load(file)

    st.subheader("Masukkan Data Untuk di Prediksi")
    open_price = st.number_input("Open Price", min_value=0.0, value=0.0)
    high_price = st.number_input("High Price", min_value=0.0, value=0.0)
    low_price = st.number_input("Low Price", min_value=0.0, value=0.0)

    if st.button("Prediksi !"):
        new_data = np.array([[open_price, high_price, low_price]])
        new_data_df = pd.DataFrame(new_data, columns=['Open', 'High', 'Low'])

        new_data_scaled = loaded_scaler.transform(new_data_df)
        new_data_poly = loaded_poly.transform(new_data_scaled)

        new_predictions = loaded_model.predict(new_data_poly)

        st.subheader("Hasil Prediksi")
        st.write(f"Prediksi Harga Penutupan: {new_predictions[0]:.2f}")
    else:
        st.subheader("Hasil Prediksi")
        st.write("Prediksi Harga Penutupan: 0.00")

