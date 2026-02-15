import streamlit as st
import pandas as pd
import joblib

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

import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="centered")
st.title("Machine Learning Classification Models")

# -------------------------------------------------
# Official test data download
# -------------------------------------------------
st.subheader("Official Test Dataset")

TEST_DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Rajeshwari-2025aa05647/ML_Assignment_2/main/test_data.csv"
)

st.markdown(
    f"[⬇️ Download Official Test Data CSV]({TEST_DATA_URL})",
    unsafe_allow_html=True
)

# -------------------------------------------------
# Model selection
# -------------------------------------------------
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

# -------------------------------------------------
# Optional upload section
# -------------------------------------------------
st.subheader("Provide Test Data for Evaluation")

uploaded_file = st.file_uploader(
    "Upload CSV test file (must contain 'num' column)",
    type=["csv"]
)

use_uploaded = st.checkbox(
    "Use uploaded CSV instead of official test data",
    value=False
)

# -------------------------------------------------
# Run evaluation ONLY on user action
# -------------------------------------------------
run_evaluation = False

if use_uploaded and uploaded_file is not None:
    if st.button("Run Evaluation on Uploaded Test Data"):
        df = pd.read_csv(uploaded_file)
        run_evaluation = True
        st.info("Evaluating on uploaded test dataset")

elif not use_uploaded:
    if st.button("Run Evaluation on Official Test Data"):
        df = pd.read_csv(TEST_DATA_URL)
        run_evaluation = True
        st.info("Evaluating on official test dataset (matches notebook & README)")

if not run_evaluation:
    st.warning(
        "Please upload test data or choose the official test dataset, "
        "then click the evaluation button."
    )
    st.stop()

# -------------------------------------------------
# Validate target
# -------------------------------------------------
if "num" not in df.columns:
    st.error("Target column 'num' not found in dataset.")
    st.stop()

# -------------------------------------------------
# Split features & target
# -------------------------------------------------
y = (df["num"] > 0).astype(int)
X = df.drop("num", axis=1)

# -------------------------------------------------
# Encode categorical features
# -------------------------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X
