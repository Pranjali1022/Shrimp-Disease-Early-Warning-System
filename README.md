# Shrimp Disease Early Warning System
Streamlit App powered by XGBoost & Scikit-learn Pipeline

This project implements a machine learning–based early warning system to predict shrimp disease risk using water quality parameters.
The system integrates data preprocessing and model inference into a single pipeline and is deployed as an interactive Streamlit web application.

# Project Overview

Shrimp aquaculture is highly sensitive to water quality fluctuations.
This application helps farmers and researchers identify disease outbreak risk in advance, enabling proactive farm management decisions.

# Key capabilities:

Accepts real-time water quality inputs

Predicts disease risk level using a trained XGBoost classifier

Provides actionable preventive recommendations

Ensures training–inference consistency via a scikit-learn Pipeline

# Machine Learning Approach

Model: XGBoost Classifier

Pipeline Components:

Missing value imputation

Feature scaling

Disease risk classification

# Output:

Class probability (predict_proba)

Risk categorization based on confidence thresholds

All preprocessing and inference steps are encapsulated inside a single serialized pipeline, eliminating manual preprocessing during deployment.

# Input Features

The model uses the following water quality parameters:

Temperature

Turbidity

Dissolved Oxygen (DO)

Biological Oxygen Demand (BOD)

CO₂

pH

Alkalinity

Hardness

Calcium

Ammonia

Nitrite

Phosphorus

H₂S

Plankton Count

# Risk Interpretation
Confidence Score	Risk Level
< 40%	Low Risk
40% – 70%	Moderate Risk
> 70%	High Risk
> 
# Each risk level is accompanied by recommended farm management actions.

# Project Structure
shrimp-disease-early-warning-system/
├── artifacts/
│   └── water_quality_XGBoost.pkl
├── app.py
├── prediction.py
├── requirements.txt
└── README.md


app.py → Streamlit user interface

prediction.py → Model inference logic

artifacts/ → Serialized ML pipeline

requirements.txt → Deployment dependencies

# Deployment

This application is deployed using Streamlit Community Cloud.

To run locally:
pip install -r requirements.txt
streamlit run app.py

# Technologies Used in the project

Python

Streamlit

Scikit-learn

XGBoost

Pandas, NumPy

Joblib

# Use Case

Early detection of shrimp disease risk

Decision support for aquaculture health management

Educational and research-oriented ML deployment example

# Key Highlights

End-to-end ML pipeline deployment

Production-safe preprocessing & inference

Modular and scalable application design

Real-world aquaculture domain application
