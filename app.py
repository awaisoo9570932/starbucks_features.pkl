import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Model aur Features ko load karein
@st.cache_resource
def load_model():
    model = joblib.load("starbucks_model.pkl")
    features = joblib.load("starbucks_features.pkl")
    return model, features

try:
    model, features = load_model()
    
    # 2. Web App ka Title
    st.title("☕ Starbucks Price Predictor")
    st.write("Enter the details below to predict the price.")

    # 3. Dynamic Inputs (Aapke features ke mutabiq)
    input_data = {}
    
    # Har feature ke liye ek numeric input field banana
    for col in features:
        # Agar features categories hain to unhein handle karne ke liye (Aap isko modify bhi kar sakte hain)
        input_data[col] = st.number_input(f"Enter value for {col}", value=0.0)

    # 4. Prediction Button
    if st.button("Predict Price"):
        # Input ko DataFrame mein convert karna
        input_df = pd.DataFrame([input_data])
        
        # Prediction karna
        prediction = model.predict(input_df)
        
        # Result dikhana
        st.success(f"💰 Predicted Price: {prediction[0]:.2f}")

except Exception as e:
    st.error(f"Error loading model or features: {e}")
    st.info("Ensure 'starbucks_model.pkl' and 'starbucks_features.pkl' are in the same folder.")
