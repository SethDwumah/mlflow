# iris_app.py
import streamlit as st
import requests
import os
import pandas as pd

st.set_page_config(page_title="Iris Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Classifier")
st.write("Enter flower measurements and get the predicted species!")

# Input fields
sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, value=5.9)
sepal_width = st.number_input("Sepal width (cm)", min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, value=3.1)
petal_width = st.number_input("Petal width (cm)", min_value=0.0, max_value=10.0, value=1.8)

if st.button("Predict"):
    # Prepare request payload
    data = {
        "columns": ["sepal length (cm)", "sepal width (cm)",
                    "petal length (cm)", "petal width (cm)"],
        "data": [[sepal_length, sepal_width, petal_length, petal_width]]
    }

    try:
        # Send request to FastAPI backend
        res = requests.post("http://127.0.0.1:8001/predict", json=data)
        if res.status_code == 200:
            result = res.json()
            st.success(f"Prediction: **{result['species'][0]}** 🌱")
            #st.write("Raw output:", result)
        else:
            st.error(f"Error {res.status_code}: {res.text}")
    except Exception as e:
        st.error(f"Could not connect to API: {e}")

st.sidebar.title("🔍 Monitoring")
if st.sidebar.button("View Logs"):
    if os.path.exists("logs/predictions.csv"):
        logs = pd.read_csv("logs/predictions.csv")
        st.subheader("📊 Prediction Logs")
        st.dataframe(logs.tail(20))  # show latest 20 predictions
    else:
        st.warning("No logs available yet.")
