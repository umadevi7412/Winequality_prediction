import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
scaler = pickle.load(open("scaler_model.csv", "rb"))
model = pickle.load(open("finalized_RFmodel.sav", "rb"))

# ---------------- CUSTOM LIGHT THEME ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #f3e7ff, #e6f0ff);
}

.title {
    text-align:center;
    font-size:40px;
    color:#6a0dad;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#555;
}

.stButton>button {
    background-color:#cdb4ff;
    color:black;
    border-radius:12px;
    height:3em;
    width:100%;
    font-size:18px;
    font-weight:bold;
}
s
.stButton>button:hover {
    background-color:#b8c0ff;
    color:white;
}

.result-box {
    background-color:#ffffff;
    padding:20px;
    border-radius:15px;
    text-align:center;
    box-shadow: 0px 0px 15px #dcdcdc;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="title">🍷 Wine Quality Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Check your wine quality instantly</p>', unsafe_allow_html=True)

st.divider()

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
    volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
    citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
    residual_sugar = st.number_input("Residual Sugar", 0.0, 15.0, 0.6)
    chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.09)
    free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0.0, 100.0, 15.0)

with col2:
    total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0.0, 300.0, 46.0)
    density = st.number_input("Density", 0.0, 2.0, 0.99)
    ph = st.number_input("pH", 0.0, 14.0, 3.3)
    sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.6)
    alcohol = st.number_input("Alcohol", 0.0, 20.0, 10.0)

st.divider()

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Check Wine Quality"):

    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, ph,
                            sulphates, alcohol]])

    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    # Convert quality to percentage
    quality_percent = (prediction / 10) * 100

    st.balloons()

    st.markdown(f"""
    <div class="result-box">
        <h2>🍷 Wine Quality Score</h2>
        <h1 style="color:#6a0dad;">{round(prediction)}</h1>
        <h3>Quality Level: {round(quality_percent,2)} %</h3>
    </div>
    """, unsafe_allow_html=True)
