import streamlit as st
import numpy as np
import pickle

# Load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Page config
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="centered")

# Styling
st.markdown("""
    <style>
        .main-title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
        }
        .sub-title {
            font-size: 20px;
            text-align: center;
            color: #777;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #FF5722;
            color: white;
            font-size: 18px;
            padding: 10px 30px;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #E64A19;
        }
        .prediction-box {
            background-color: #e8f5e9;
            border: 2px solid #4CAF50;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-top: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-title">🌾 Crop Yield Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter the features below and get the predicted crop yield per country</div>', unsafe_allow_html=True)

# Input Form
with st.form("predict_form"):
    Area = st.text_input("📍 Area (Country)")
    Item = st.text_input("🌽 Item (Crop Name)")
    Year = st.number_input("📅 Year", min_value=1900, max_value=2100, step=1)
    average_rain_fall_mm_per_year = st.number_input("🌧️ Average Rainfall (mm/year)", step=0.1)
    pesticides_tonnes = st.number_input("🧪 Pesticides Used (tonnes)", step=0.1)
    avg_temp = st.number_input("🌡️ Average Temperature (°C)", step=0.1)

    submitted = st.form_submit_button("Predict Yield")

# Prediction Logic
if submitted:
    try:
        features = np.array([[Area, Item, Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp]], dtype=object)
        transformed = preprocessor.transform(features)
        prediction = dtr.predict(transformed)[0]

        st.markdown(f"""
            <div class="prediction-box">
                <h2>✅ Predicted Yield: <span style="color:#2E7D32">{prediction:.2f}</span> hg/ha</h2>
            </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Error: {e}")
