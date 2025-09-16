import streamlit as st
import numpy as np
import joblib

model = joblib.load("crop_prediction_model.pkl")
le = joblib.load("label_encoder.pkl")

st.set_page_config(
    page_title="🌱 KrishiVerse - Crop Recommendation",
    page_icon="🌾",
    layout="centered"
)

st.markdown(
    """
    <div style="text-align:center">
        <h1 style="color:#2e7d32;">🌱 KrishiVerse</h1>
        <p style="font-size:18px;color:#4caf50;">Crop Recommendation System - Enter soil and weather parameters below</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0, max_value=140, value=50, step=1)
    P = st.number_input("Phosphorus (P)", min_value=5, max_value=145, value=50, step=1)
    K = st.number_input("Potassium (K)", min_value=5, max_value=205, value=50, step=1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

with col2:
    temperature = st.number_input("Temperature (°C)", min_value=8.0, max_value=45.0, value=25.0, step=0.1, format="%.1f")
    humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0, value=50.0, step=0.1, format="%.1f")
    ph = st.number_input("pH value", min_value=3.5, max_value=9.0, value=6.5, step=0.1, format="%.1f")

st.write("---")

if st.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    crop_name = le.inverse_transform(prediction)[0]

    st.markdown(
        f"""
        <div style="background-color:#e8f5e9;padding:20px;border-radius:10px">
            <h2 style="color:#2e7d32;text-align:center;">🌿 Recommended Crop</h2>
            <h1 style="text-align:center;color:#1b5e20;">{crop_name}</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    descriptions = {
        "rice": "Rice grows well in warm, humid climates with ample water.",
        "wheat": "Wheat prefers moderate temperature and low humidity.",
        "maize": "Maize thrives in well-drained fertile soil with moderate rainfall."
    }

    if crop_name.lower() in descriptions:
        st.markdown(
            f"""
            <div style="background-color:#fff3e0;padding:15px;border-radius:10px;margin-top:10px">
                <strong>Description:</strong> {descriptions[crop_name.lower()]}
            </div>
            """,
            unsafe_allow_html=True
        )
