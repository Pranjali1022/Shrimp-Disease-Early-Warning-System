import joblib
import os
import pandas as pd

# DEBUG: show exactly where we are and what files exist
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print("DEBUG: predict.py directory:", CURRENT_DIR)
print("DEBUG: files here:", os.listdir(CURRENT_DIR))

# Try loading model from same directory
MODEL_PATH = os.path.join(CURRENT_DIR, "model.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"model.pkl not found in {CURRENT_DIR}. Files present: {os.listdir(CURRENT_DIR)}"
    )

pipeline = joblib.load(MODEL_PATH)

def predict_water_quality(df):
    preds = pipeline.predict(df)
    label_map = {0: "Good", 1: "Caution", 2: "Danger"}
    df["Status"] = [label_map[p] for p in preds]
    return df
