# Shrimp Water Quality Prediction (ML + Streamlit)

This project predicts shrimp pond water quality using a Machine Learning model and preprocessing pipeline.  
Users can upload a CSV file with water parameters to get:

- Water quality status (**Good / Caution / Danger**)
- Suggestions to improve water conditions

The model is saved as a **single pipeline (`model.pkl`)**, which includes:
- Imputer (handles missing values)
- Scaler (StandardScaler)
- Trained ML classifier

---

## 🚀 How to Run the App Locally

### 1. Install required libraries  
Open terminal inside your project folder and run:

```bash
pip install -r requirements.txt
```

### 2. Start the Streamlit app

```bash
streamlit run app.py
```

This will open:

```
http://localhost:8501
```

---

## 📁 Project Structure

```
shrimp_water_quality_model/
├── model.pkl          # ML pipeline (imputer + scaler + trained model)
├── predict.py         # Prediction code
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## 📥 CSV Input Format

Your input CSV **must include the same columns as the training dataset**.

Example:

```csv
pH,do,bod,turbidity,nitrate
7.2,6.5,3.1,4.5,2.0
6.0,2.5,9.0,30,45
8.8,5.0,4.0,15,25
```

Column names are case-sensitive.

---

## 🧠 Label Mapping

```
0 → Good
1 → Caution
2 → Danger
```

These labels appear as text in the results table in Streamlit.

---

## 🛠 How Prediction Works

When you upload a CSV:

1. The Streamlit app loads `model.pkl`
2. The pipeline:
   - Imputes missing values
   - Scales features
   - Runs predictions using the ML model
3. The app displays:
   - Status per row
   - Suggestions based on prediction

---

## 📌 Additional Notes

- You can retrain the model in Google Colab and overwrite `model.pkl`.
- Make sure your CSV columns match exactly.
- The project is designed for easy deployment on **Streamlit Cloud**.

---

## 👩‍💻 Author
**Pranjali Patil**

---
