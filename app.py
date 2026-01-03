import streamlit as st
import pandas as pd
from predict import predict_water_quality

st.title("Shrimp Water Quality Prediction App")
st.write("Upload water quality data to predict status and get suggestions.")

uploaded = st.file_uploader("Upload CSV file", type="csv")

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Input Data")
    st.dataframe(df)

    # Run prediction
    result = predict_water_quality(df)

    st.write("Prediction Results")
    st.dataframe(result)

    st.success("Prediction completed successfully!")
