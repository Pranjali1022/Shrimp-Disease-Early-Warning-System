import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "artifacts/water_quality_XGBoost.pkl"

def predict_risk(input_df):
    obj = joblib.load(MODEL_PATH)
    model = obj["model"]
    imputer = obj["imputer"]
    scaler = obj["scaler"]
    features = obj["features"]

    X_imp = imputer.transform(input_df[features])
    X_scaled = scaler.transform(X_imp)
    probs = model.predict_proba(X_scaled)[0]

    return probs

