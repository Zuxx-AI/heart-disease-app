import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")

folder = Path(__file__).parent
model = pickle.load(open(folder / "svc_heart_model.pkl", "rb"))
scaler = pickle.load(open(folder / "age_scaler.pkl", "rb"))

sex_encoder = LabelEncoder().fit(["Male", "Female"])
chestpain_encoder = LabelEncoder().fit(["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
vessels_encoder = LabelEncoder().fit(["Zero", "One", "Two", "Three"])

st.title("Heart Disease Prediction System")
st.write("Enter the patient details and the model will predict whether heart disease is likely. "
         "This is a student project and not medical advice.")

sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 25, 80, 55)
chest_pain = st.selectbox("Chest pain type", ["Typical", "Atypical", "NonAnginal", "Asymptomatic"])
vessels = st.selectbox("Major vessels", ["Zero", "One", "Two", "Three"])

if st.button("Predict"):
    features = pd.DataFrame({
        "Sex": sex_encoder.transform([sex]),
        "Age": scaler.transform(pd.DataFrame({"Age": [age]})).ravel(),
        "ChestPain": chestpain_encoder.transform([chest_pain]),
        "MajorVessels": vessels_encoder.transform([vessels]),
    })
    result = model.predict(features)[0]
    if result == 1:
        st.error("HEART DISEASE LIKELY")
    else:
        st.success("NO HEART DISEASE")
    st.write("Encoded values sent to the model:")
    st.dataframe(features, hide_index=True)