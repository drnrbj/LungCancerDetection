import streamlit as st
import pandas as pd
import joblib
import numpy as np

model = joblib.load("lungcancer_svm.pkl")
scaler = joblib.load("lungcancer_scaler.pkl")

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("Lung Cancer Detection")
st.write("Fill in the following information to predict the likelihood of lung cancer.")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=50)
smoking = st.selectbox("Do you smoke?", ["No", "Yes"])
yellow_fingers = st.selectbox("Yellow fingers?", ["No", "Yes"])
anxiety = st.selectbox("Anxiety?", ["No", "Yes"])
peer_pressure = st.selectbox("Peer pressure?", ["No", "Yes"])
chronic_disease = st.selectbox("Chronic disease?", ["No", "Yes"])
fatigue = st.selectbox("Fatigue?", ["No", "Yes"])
allergy = st.selectbox("Allergy?", ["No", "Yes"])
wheezing = st.selectbox("Wheezing?", ["No", "Yes"])
alcohol_consumption = st.selectbox("Alcohol consumption?", ["No", "Yes"])
coughing = st.selectbox("Coughing?", ["No", "Yes"])
shortness_of_breath = st.selectbox("Shortness of breath?", ["No", "Yes"])
swallowing_difficulty = st.selectbox("Swallowing difficulty?", ["No", "Yes"])
chest_pain = st.selectbox("Chest pain?", ["No", "Yes"])

input_data = pd.DataFrame({
    "GENDER": [0 if gender=="Male" else 1],
    "AGE": [age],
    "SMOKING": [0 if smoking=="No" else 1],
    "YELLOW_FINGERS": [0 if yellow_fingers=="No" else 1],
    "ANXIETY": [0 if anxiety=="No" else 1],
    "PEER_PRESSURE": [0 if peer_pressure=="No" else 1],
    "CHRONIC DISEASE": [0 if chronic_disease=="No" else 1],
    "FATIGUE ": [0 if fatigue=="No" else 1],
    "ALLERGY ": [0 if allergy=="No" else 1],
    "WHEEZING": [0 if wheezing=="No" else 1],
    "ALCOHOL CONSUMING": [0 if alcohol_consumption=="No" else 1],
    "COUGHING": [0 if coughing=="No" else 1],
    "SHORTNESS OF BREATH": [0 if shortness_of_breath=="No" else 1],
    "SWALLOWING DIFFICULTY": [0 if swallowing_difficulty=="No" else 1],
    "CHEST PAIN": [0 if chest_pain=="No" else 1]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction[0] == 1:
        st.error(f"Lung Cancer Detected! Probability: {probability:.2f}")
    else:
        st.success(f"No Lung Cancer Detected. Probability: {probability:.2f}")
