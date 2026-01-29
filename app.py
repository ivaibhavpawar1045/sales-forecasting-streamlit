import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("Sales Forecasting App")
st.write("Forecast future sales using ARIMA time-series model")

@st.cache_data
def load_data():
    df = pd.read_csv("data/sales.csv", encoding="latin1")
    df.columns = df.columns.str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df = df.groupby("Order Date")["Sales"].sum().reset_index()
    df = df.sort_values("Order Date")
    return df


df = load_data()

st.subheader("Historical Sales Data")
st.line_chart(df.set_index("Order Date"))

# Train ARIMA
model = ARIMA(df["Sales"], order=(5,1,0))
model_fit = model.fit()

# Forecast
n_periods = st.slider("Days to Forecast", 7, 60, 30)
forecast = model_fit.forecast(steps=n_periods)

future_dates = pd.date_range(
    start=df["Order Date"].iloc[-1],
    periods=n_periods+1,
    freq="D"
)[1:]

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecast Sales": forecast
})

st.subheader("Sales Forecast")
st.line_chart(forecast_df.set_index("Date"))

st.dataframe(forecast_df)
