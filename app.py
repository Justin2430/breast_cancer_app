# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the saved scaler and model
def load_artifacts(scaler_file="models/scaler.pkl", model_file="models/ann_model.pkl"):
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return scaler, model

# Streamlit App
def main():
    st.title("Breast Cancer Prediction")
    st.write("""
    This app predicts whether a tumor is **Malignant** or **Benign** based on user inputs for selected features.
    """)

    # Load artifacts
    scaler, model = load_artifacts()

    # Define the selected features (from feature selection step)
    selected_features = [
        "mean radius", "mean texture", "mean smoothness", "mean compactness",
        "mean concavity", "mean concave points", "mean symmetry",
        "mean fractal dimension", "radius error", "texture error"
    ]

    # User input
    st.subheader("Enter Feature Values:")
    user_input = []
    for feature in selected_features:
        value = st.number_input(f"{feature}", value=0.0)
        user_input.append(value)

    # Predict based on user input
    if st.button("Predict"):
        # Convert user input to a NumPy array and scale it
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)
        result = "Malignant" if prediction[0] == 0 else "Benign"

        # Display the result
        st.subheader("Prediction:")
        st.write(f"The tumor is predicted to be **{result}**.")

if __name__ == "__main__":
    main()
