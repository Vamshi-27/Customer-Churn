import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load files
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

st.title("Customer Churn Prediction")

st.write("Fill customer details below:")

input_data = {}

# 🔥 Dynamically create UI (clean + correct values)
for col in columns:
    
    if col in encoders:  # categorical
        input_data[col] = st.selectbox(col, encoders[col].classes_)
    
    else:  # numerical
        if col == "tenure":
            input_data[col] = st.slider(col, 0, 72, 12)
        else:
            input_data[col] = st.number_input(col, value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Encode properly
for col in input_df.columns:
    if col in encoders:
        le = encoders[col]
        input_df[col] = input_df[col].apply(lambda x: le.transform([x])[0])

# Ensure correct order
input_df = input_df.reindex(columns=columns)

# Scale
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer likely to CHURN (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Customer likely to STAY (Probability: {prob:.2f})")