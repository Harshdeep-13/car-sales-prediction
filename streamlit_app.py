"""
streamlit_app.py — Streamlit version of the car resale price predictor.

Run locally:
    streamlit run streamlit_app.py
"""

import json
import pickle
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Resale Value Estimator",
    page_icon="🚗",
    layout="centered",
)

# ---------------------------------------------------------------
# Load model + metadata (cached so it only loads once per session)
# ---------------------------------------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model_metadata.json", "r") as f:
        metadata = json.load(f)
    return model, metadata

model, metadata = load_model()
CATEGORICAL_VALUES = metadata["categorical_values"]
CURRENT_YEAR = datetime.now().year

# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🚗 Resale Value Estimator")
st.caption(
    f"Linear Regression model · trained on 299 listings · "
    f"R² ≈ {metadata['metrics']['r2']:.2f}, MAE ≈ ₹{metadata['metrics']['mae']:.2f} lakh"
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input(
            "Registration Year", min_value=1990, max_value=CURRENT_YEAR, value=2018
        )
        driven_kms = st.number_input(
            "Kilometres Driven", min_value=0, value=35000, step=1000
        )
        present_price = st.number_input(
            "Current Ex-Showroom Price (₹ Lakhs)",
            min_value=0.0, value=8.5, step=0.1, format="%.2f"
        )
    with col2:
        fuel_type = st.selectbox("Fuel Type", CATEGORICAL_VALUES["Fuel_Type"])
        transmission = st.selectbox("Transmission", CATEGORICAL_VALUES["Transmission"])
        selling_type = st.selectbox("Seller Type", CATEGORICAL_VALUES["Selling_type"])
        owner = st.selectbox("Previous Owners", [0, 1, 2, 3])

    submitted = st.form_submit_button("Estimate Selling Price", use_container_width=True)

if submitted:
    car_age = CURRENT_YEAR - year
    row = pd.DataFrame([{
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner,
        "Car_Age": car_age,
    }])

    prediction = model.predict(row)[0]
    prediction = max(0.0, round(float(prediction), 2))

    st.success("Estimate ready")
    m1, m2 = st.columns(2)
    m1.metric("Estimated Selling Price", f"₹{prediction} L")
    m2.metric("In Rupees", f"₹{prediction * 100000:,.0f}")

    with st.expander("See the exact inputs sent to the model"):
        st.dataframe(row, use_container_width=True)

st.divider()
st.caption(
    "Model: scikit-learn Pipeline (OneHotEncoder + LinearRegression), "
    "features: Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner, Car_Age."
)
