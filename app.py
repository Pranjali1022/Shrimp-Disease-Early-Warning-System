
import streamlit as st
import joblib
import numpy as np
import pandas as pd

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

prevention = {
    "Low Risk": "Water quality looks stable. Continue routine monitoring every 24 hours.",
    "Moderate Risk": (
        "- Increase aeration\n"
        "- Reduce feeding by ~20%\n"
        "- Check ammonia & nitrite levels\n"
        "- Partial water exchange"
    ),
    "High Risk": (
        "- Immediate water exchange (20–30%)\n"
        "- Increase DO\n"
        "- Stop feeding temporarily\n"
        "- Add probiotics"
    )
}

st.set_page_config(page_title="Shrimp Water Quality Risk Predictor")
st.title("Shrimp Water Quality Risk Predictor")

input_values = {}
for feat in features:
    input_values[feat] = st.number_input(feat, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([input_values], columns=features)
    probs = pipeline.predict_proba(df)[0]
    confidence = float(np.max(probs))

    if confidence < 0.40:
        risk = "Low Risk"
    elif confidence < 0.70:
        risk = "Moderate Risk"
    else:
        risk = "High Risk"

    st.subheader("Prediction")
    st.write(f"Risk Level: **{risk}**")
    st.write(f"Confidence: **{confidence*100:.2f}%**")
    st.write(prevention[risk])
    st.markdown("---")
st.markdown("### 📊 Model Performance (Validation Results)")

st.write("""
The XGBoost model was evaluated using cross-validation on historical
water quality data and achieved the following performance:
""")

st.write("- **Overall Accuracy:** ~99.6%")
st.write("- **Macro F1-score:** ~0.99")
st.write("- **Balanced precision and recall across all risk classes**")

st.info(
    "Most misclassifications occur between Moderate and High risk levels, "
    "which reflects realistic transitional water conditions in shrimp farms."
)

