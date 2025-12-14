import streamlit as st
import pandas as pd
import joblib

rf_model = joblib.load("lungcancer_rf.pkl")

st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("Lung Cancer Detection")
st.write("Fill in the following information to predict the likelihood of lung cancer.")

st.markdown("""
    <style>
    div[data-baseweb="radio"] label {
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="input"] label {
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    
    div[data-baseweb="button"] {
        font-size: 18px !important;
    }
    </style> """, unsafe_allow_html=True)

gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
age = st.number_input("Age", min_value=0, max_value=120, value=50)
smoking = st.radio("Do you smoke?", ["No", "Yes"], horizontal=True)
yellow_fingers = st.radio("Yellow fingers?", ["No", "Yes"], horizontal=True)
anxiety = st.radio("Anxiety?", ["No", "Yes"], horizontal=True)
peer_pressure = st.radio("Peer pressure?", ["No", "Yes"], horizontal=True)
chronic_disease = st.radio("Chronic disease?", ["No", "Yes"], horizontal=True)
fatigue = st.radio("Fatigue?", ["No", "Yes"], horizontal=True)
allergy = st.radio("Allergy?", ["No", "Yes"], horizontal=True)
wheezing = st.radio("Wheezing?", ["No", "Yes"], horizontal=True)
alcohol_consumption = st.radio("Alcohol consumption?", ["No", "Yes"], horizontal=True)
coughing = st.radio("Coughing?", ["No", "Yes"], horizontal=True)
shortness_of_breath = st.radio("Shortness of breath?", ["No", "Yes"], horizontal=True)
swallowing_difficulty = st.radio("Swallowing difficulty?", ["No", "Yes"], horizontal=True)
chest_pain = st.radio("Chest pain?", ["No", "Yes"], horizontal=True)

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

if st.button("Predict", use_container_width=True, 
             type="primary", 
             help="Click to get prediction"):
    prediction = rf_model.predict(input_data)
    probability = rf_model.predict_proba(input_data)[0][1]
    
    st.markdown(f"""
        <div style='text-align: center; padding: 20px;'>
            <h2>Prediction Result</h2>
        """, unsafe_allow_html=True)
    
    if probability > 0.3:
        st.error(f"🚨 **Lung Cancer Detected!**  \n**Probability:** {probability:.2%}")
    else:
        st.success(f"✅ **No Lung Cancer Detected**  \n**Probability:** {probability:.2%}")