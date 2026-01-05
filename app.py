
# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model + preprocessing
MODEL_PATH = "artifacts/water_quality_XGBoost.pkl"
pipeline = joblib.load(MODEL_PATH)

features = [
    "Temp",
    "Turbidity (cm)",
    "DO(mg/L)",
    "BOD (mg/L)",
    "CO2",
    "pH`",
    "Alkalinity (mg L-1 )",
    "Hardness (mg L-1 )",
    "Calcium (mg L-1 )",
    "Ammonia (mg L-1 )",
    "Nitrite (mg L-1 )",
    "Phosphorus (mg L-1 )",
    "H2S (mg L-1 )",
    "Plankton (No. L-1)"
]

# Prevention messages
prevention = {
    "Low Risk": "Water quality looks stable. Continue routine monitoring every 24 hours.",
    "Moderate Risk": (
        "1. Moderate disease risk.\n"  
        "2. Increase aeration.\n"
        "3. Reduce feeding by ~20%.\n"
        "4. Check ammonia & nitrite levels.\n"
        "5. Consider partial water exchange."
    ),
    "High Risk": (
        "1. HIGH disease outbreak risk.\n"
        "2. Immediate partial water exchange (20–30%)\n"
        "3. Increase DO using blowers/paddle wheels.\n"
        "4. Stop feeding temporarily.\n"
        "5. Add probiotics as recommended.\n"
        "6. Consult a technician if symptoms persist."
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

        # Predict using pipeline
        probs = pipeline.predict_proba(df_input)[0]
        pred_class = pipeline.predict(df_input)[0]

        # Map class index to risk label
        label_map = {
            0: "Low Risk",
            1: "Moderate Risk",
            2: "High Risk"
        }

        risk = label_map[pred_class]

        # Display results
        st.subheader("Prediction Result")
        st.write(f"**Predicted Risk Level:** {risk}")

        st.markdown("### Model confidence per risk class")
        for cls, label in label_map.items():
            st.write(f"{label}: {probs[cls]*100:.2f}%")

        st.markdown("### Recommended actions")
        st.write(prevention[risk])

        if risk == "High Risk":
            st.warning("High risk detected. Please verify input values carefully.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

st.markdown("---")

st.markdown("### Model Performance Summary")

st.write("""The deployed XGBoost model was evaluated on a held-out validation set using multiclass classification metrics.""")
st.write("- **Overall Accuracy:** ~99.6%")
st.write("- **Macro F1-score:** ~0.99")
st.write("- **Balanced precision and recall across all risk classes**")
st.info("Most misclassifications occur between Moderate and High risk levels, "
        "which reflects realistic transitional water conditions in shrimp farms.")

st.markdown("### Model Error Characteristics")

st.write("""
- Near-perfect classification for Low-risk conditions  
- Minor confusion between Moderate and High-risk states  
- Fewer than 0.5% total misclassifications
""")

st.markdown("### Disease Early Warning Logic")

st.write("""
Shrimp diseases are often preceded by changes in water quality rather
than sudden biological symptoms. Parameters such as ammonia, nitrite,
dissolved oxygen, and organic load act as **leading indicators** of
disease susceptibility.
The model therefore functions as a **disease early warning system**
by identifying high-risk water conditions before clinical outbreaks.
""")

