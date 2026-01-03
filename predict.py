import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
pipeline = joblib.load(model_path)

def predict_water_quality(df):
    # Predictions
    preds = pipeline.predict(df)

    # Convert numeric output to text labels
    label_map = {0: "Good", 1: "Caution", 2: "Danger"}
    df["Status"] = [label_map[p] for p in preds]

    # Simple suggestions based on status
    df["Suggestion"] = df["Status"].apply(lambda s:
        "Water quality is good — no action needed." if s == "Good" else
        "Minor changes needed. Monitor water parameters." if s == "Caution" else
        "High risk! Immediate corrective action required."
    )

    return df
