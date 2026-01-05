
# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model + preprocessing
MODEL_PATH = "artifacts/water_quality_XGBoost.pkl"

try:
    obj = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(f"Model file not found at {MODEL_PATH}. Please check path.")
    st.stop()

model = obj["model"]
imputer = obj["imputer"]
scaler = obj["scaler"]
features = obj["features"]

# Prevention messages
prevention = {
    "Low Risk": "Water quality looks stable. Continue routine monitoring every 24 hours.",
    "Moderate Risk": (
        "- Moderate disease risk.
"
        "- Increase aeration
"
        "- Reduce feeding by ~20%
"
        "- Check ammonia & nitrite levels
"
        "- Consider partial water exchange"
    ),
    "High Risk": (
        "- HIGH disease outbreak risk!
"
        "- Immediate partial water exchange (20–30%)
"
        "- Increase DO using blowers/paddle wheels
"
        "- Stop feeding temporarily
"
        "- Add probiotics as recommended
"
        "- Consult a technician if symptoms persist"
    )
}

# Page title
st.set_page_config(page_title="Water Quality Risk Predictor for Shrimp Pond")
st.title("Water Quality Risk Predictor for Shrimp Pond")
st.write("Enter water-parameter values below and click Predict.")

default_values = {
    "Temp": 30.2,
        "Turbidity (cm)": 5.0,
        "DO(mg/L)": 7.5,
        "BOD (mg/L)": 2.0,
        "CO2": 15.0,
        "pH`": 7.2,
        "Alkalinity (mg L-1 )": 100.0,
        "Hardness (mg L-1 )": 150.0,
        "Calcium (mg L-1 )": 50.0,
        "Ammonia (mg L-1 )": 0.1,
        "Nitrite (mg L-1 )": 0.01,
        "Phosphorus (mg L-1 )": 0.05,
        "H2S (mg L-1 )": 0.001,
        "Plankton (No. L-1)": 1000.0
}

with st.form(key="input_form"):
    input_values = {}
    for feat in features:
        input_values[feat] = st.number_input(
            label=str(feat),
            value=float(default_values.get(feat, 1.0)),
            format="%.4f"
        )
    submit = st.form_submit_button("Predict Disease Risk")

# Prediction logic
if submit:
    try:
        # Build DataFrame
        df_input = pd.DataFrame([input_values], columns=features)

        # Preprocess
        X_imp = imputer.transform(df_input)
        X_scaled = scaler.transform(X_imp)

        # Predict
        probs = model.predict_proba(X_scaled)[0]
        pred_class = model.predict(X_scaled)[0]

        # Class → label mapping
        label_map = {
            0: "Low Risk",
            1: "Moderate Risk",
            2: "High Risk"
        }

        risk = label_map[pred_class]

        # ----------------------------
        # Display results
        # ----------------------------
        st.subheader("Prediction Result")
        st.write(f"**Predicted Risk Level:** {risk}")

        st.markdown("Model confidence per risk class")
        for cls, label in label_map.items():
            st.write(f"{label}: {probs[cls]*100:.2f}%")

        st.markdown("Recommended actions")
        st.write(prevention[risk])

        if risk == "High Risk":
            st.warning("High risk detected. Please verify input values carefully.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
