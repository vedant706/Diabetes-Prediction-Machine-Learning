import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved model and scaler
with open('diabetes_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Page configuration
st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="centered")

# Custom Title for College Submission
st.title("🩺 Diabetes Prediction System")
st.markdown("**Developed by: Vedant Tarale** | ML Mini Project")
st.markdown("---")

st.write("Enter the patient's medical details below to predict the likelihood of diabetes.")

# Create layout with columns for cleaner UI
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120, step=1)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70, step=1)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20, step=1)

with col2:
    insulin = st.number_input("Insulin Level (IU/mL)", min_value=0, max_value=900, value=79, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age (Years)", min_value=1, max_value=120, value=33, step=1)

# Prediction Button
if st.button("Predict Diabetes Status", type="primary"):
    # Group inputs into a numpy array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Scale the input data using the saved scaler
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    prediction_proba = model.predict_proba(scaled_input)
    
    st.markdown("---")
    
    # Display results
    if prediction[0] == 1:
        st.error(f"⚠️ The model predicts that the patient **IS DIABETIC**.")
    else:
        st.success(f"✅ The model predicts that the patient **IS NOT DIABETIC**.")
        
    st.info(f"**Confidence Score:** {np.max(prediction_proba) * 100:.2f}%")