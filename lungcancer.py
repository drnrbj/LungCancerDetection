import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load("lungcancer_rf.pkl")

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
    "SMOKING": [1 if smoking=="Yes" else 0],
    "YELLOW_FINGERS": [1 if yellow_fingers=="Yes" else 0],
    "ANXIETY": [1 if anxiety=="Yes" else 0],
    "PEER_PRESSURE": [1 if peer_pressure=="Yes" else 0],
    "CHRONIC DISEASE": [1 if chronic_disease=="Yes" else 0],
    "FATIGUE ": [1 if fatigue=="Yes" else 0],
    "ALLERGY ": [1 if allergy=="Yes" else 0],
    "WHEEZING": [1 if wheezing=="Yes" else 0],
    "ALCOHOL CONSUMING": [1 if alcohol_consumption=="Yes" else 0],
    "COUGHING": [1 if coughing=="Yes" else 0],
    "SHORTNESS OF BREATH": [1 if shortness_of_breath=="Yes" else 0],
    "SWALLOWING DIFFICULTY": [1 if swallowing_difficulty=="Yes" else 0],
    "CHEST PAIN": [1 if chest_pain=="Yes" else 0]
})

if st.button("Predict"):
    prediction = rf_model.predict(input_data)
    probability = rf_model.predict_proba(input_data)[0][1]

    if probability > 0.3:
        st.error(f"Lung Cancer Detected! Probability: {probability:.2f}")
    else:
        st.success(f"No Lung Cancer Detected. Probability: {probability:.2f}")

