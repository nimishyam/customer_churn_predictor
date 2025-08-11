import streamlit as st
import tensorflow as tf
import pandas as pd
import pickle
import numpy as np

# ---------------------------
# Load Model & Encoders
# ---------------------------
model = tf.keras.models.load_model('3-output/churn_model.h5')

with open('1-input/scalar.pkl', 'rb') as file:
    scalar = pickle.load(file)

with open('1-input/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('1-input/onehot_encoder_geography.pkl', 'rb') as file:
    onehot_geo = pickle.load(file)

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="AI Churn Predictor", layout="centered")

# ---------------------------
# Custom CSS for Futuristic Theme
# ---------------------------
st.markdown("""
    <style>
        .stApp { background-color: #0b0f15; color: #e6eef8; font-family: 'Segoe UI', sans-serif; }
        h1, h2, h3 { background: linear-gradient(90deg, #00d4ff, #00ff99); -webkit-background-clip: text; color: transparent; }
        
        /* Styled inputs */
        .stSelectbox, .stNumberInput, .stSlider { color: #e6eef8 !important; }
        div[data-baseweb="select"] > div { background-color: #1c1f26; color: white; border-radius: 8px; }
        input, textarea { background-color: #1c1f26 !important; color: white !important; border-radius: 6px; }
        
        /* Animated results */
        @keyframes fadeInScale { 0% {opacity:0; transform:scale(0.95);} 100% {opacity:1; transform:scale(1);} }
        @keyframes gradientMove { 0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;} }

        .result-card {
            padding: 20px; border-radius: 12px; text-align: center;
            font-size: 20px; font-weight: bold; margin-top: 20px; width: 100%;
            box-sizing: border-box; animation: fadeInScale 0.5s ease-in-out;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }
        .churn { background: linear-gradient(270deg, #ff6b6b, #ff4b4b, #ff2e63);
                 background-size: 200% 200%; color: white;
                 animation: gradientMove 3s ease infinite, fadeInScale 0.5s ease-in-out; }
        .stay { background: linear-gradient(270deg, #2ecc71, #27ae60, #1abc9c);
                background-size: 200% 200%; color: white;
                animation: gradientMove 3s ease infinite, fadeInScale 0.5s ease-in-out; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Title
# ---------------------------
st.title("AI-Powered Customer Churn Predictor")
st.markdown("Predict if a customer will churn using an AI model trained on historical data.")

# ---------------------------
# Input Form
# ---------------------------
with st.form("churn_form"):
    st.subheader("Enter Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        geography = st.selectbox('Geography', onehot_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92, 30)
        tenure = st.slider('Tenure (Years)', 0, 10, 5)
        num_of_products = st.slider('Number of Products', 1, 4, 1)
    with col2:
        credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=600)
        balance = st.number_input('Balance', min_value=0.0, value=50000.0)
        estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])
        is_active_member = st.selectbox('Active Member', [0, 1])

    submitted = st.form_submit_button("Predict Churn!")

# ---------------------------
# Prediction Logic
# ---------------------------
if submitted:
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = onehot_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_geo.get_feature_names_out(['Geography']))
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    input_data_scaled = scalar.transform(input_data)

    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    # Display result card
    if prediction_proba > 0.5:
        st.markdown(f"<div class='result-card churn'>🚨 Churn Probability: {prediction_proba:.2%} — High Risk of Churn</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-card stay'>✅ Churn Probability: {prediction_proba:.2%} — Low Risk of Churn</div>", unsafe_allow_html=True)

    # Probability bar
    st.progress(float(prediction_proba))

    # Risk category
    risk_level = "High" if prediction_proba > 0.75 else "Medium" if prediction_proba > 0.5 else "Low"
    st.write(f"📊 **Risk Category:** {risk_level}",layout="centered")
    
    # Save to session history
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({"Geography": geography, "Gender": gender, "Probability": prediction_proba, "Risk": risk_level})

# ---------------------------
# History Section
# ---------------------------
if "history" in st.session_state and st.session_state.history:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))
