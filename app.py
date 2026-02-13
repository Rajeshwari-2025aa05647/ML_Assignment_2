%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="ML Assignment 2", layout="centered")
st.title("Machine Learning Classification Models")

uploaded_file = st.file_uploader("Upload CSV test file", type=["csv"])

model_name = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if "num" not in df.columns:
        st.error("Target column 'num' not found in uploaded file.")
        st.stop()

    # ----- TARGET -----
    y = (df["num"] > 0).astype(int)
    X = df.drop("num", axis=1)

    # ----- ENCODE CATEGORICAL FEATURES -----
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # ----- HANDLE MISSING VALUES -----
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # ----- SCALE FEATURES -----
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ----- LOAD MODEL -----
    model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")

    # ----- PREDICT -----
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # ----- METRICS -----
    st.subheader("Evaluation Metrics")
    st.write({
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred)
    })

    st.subheader("Confusion Matrix")
    st.write(confusion_matrix(y, y_pred))

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))
