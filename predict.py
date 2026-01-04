import os
import joblib
import pandas as pd

# Directory where predict.py is located
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(THIS_FILE_DIR, "model.pkl")

pipeline = joblib.load(MODEL_PATH)

def predict_water_quality(df):
    preds = pipeline.predict(df)

    label_map = {0: "Good", 1: "Caution", 2: "Danger"}
    df["Status"] = [label_map[p] for p in preds]

    df["Suggestion"] = df["Status"].apply(
        lambda s: "Water quality is good — no action needed."
        if s == "Good"
        else "Monitor parameters closely."
        if s == "Caution"
        else "High risk! Immediate corrective action required."
    )

    return df
