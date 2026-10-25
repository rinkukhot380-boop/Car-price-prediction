import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Used Car Price Predictor", page_icon="🚗")


@st.cache_resource
def load_files():
    model = joblib.load("car_model.pkl")
    options = joblib.load("car_options.pkl")
    return model, options


model, options = load_files()

st.title("🚗 Used Car Price Predictor")
st.write("Enter the car's details to get an estimated resale price.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", options["brand"])
    fuel = st.selectbox("Fuel type", options["fuel"])
    owner = st.selectbox("Owner", options["owner"])

with col2:
    year = st.slider("Year of manufacture", 1995, 2020, 2015)
    km_driven = st.number_input("Km driven", min_value=0, max_value=500000,
                                value=50000, step=1000)
    max_power = st.number_input("Max power (bhp)", min_value=30.0,
                                max_value=400.0, value=80.0)
    engine = st.number_input("Engine (CC)", min_value=600, max_value=5000,
                             value=1200, step=100)

if st.button("Predict price"):
    # Column names must match what the pipeline was trained on
    row = pd.DataFrame([{
        "year": year,
        "km_driven": km_driven,
        "max_power_num": max_power,
        "engine_num": engine,
        "fuel": fuel,
        "owner": owner,
        "brand": brand,
    }])

    log_price = model.predict(row)[0]
    price = np.exp(log_price)  # model predicts log(price), convert back to rupees

    st.success(f"Estimated price: ₹{price:,.0f}")
    st.caption(
        "Predictions are typically typical error about ₹74,000. "
        "The model was trained on listings up to 2020, so prices reflect "
        "that market and may be lower than today's."
    )