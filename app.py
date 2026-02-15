import streamlit as st
import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
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
# Session state initialization
# -------------------------------------------------
if "run_evaluation" not in st.session_state:
    st.session_state.run_evaluation = False

if "df_eval" not in st.session_state:
    st.session_state.df_eval = None

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
# Upload section
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
# Evaluation trigger (session-safe)
# -------------------------------------------------
if use_uploaded and uploaded_file is not None:
    if st.button("Run Evaluation on Uploaded Test Data"):
        st.session_state.df_eval = pd.read_csv(uploaded_file)
        st.session_state.run_evaluation = True
        st.success("Evaluating on uploaded test dataset")

elif not use_uploaded:
    if st.button("Run Evaluation on Official Test Data"):
        st.session_state.df_eval = pd.read_csv(TEST_DATA_URL)
        st.session_state.run_evaluation = True
        st.success("Evaluating on official test dataset (matches notebook & README)")

# -------------------------------------------------
# Stop execution until evaluation is triggered
# -------------------------------------------------
if not st.session_state.run_evaluation:
    st.info(
        "Please upload a test dataset or choose the official test dataset, "
        "then click the evaluation button."
    )
    st.stop()

df = st.session_state.df_eval

# -------------------------------------------------
# Validate target column
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
# Encode categorical features (same as notebook)
# -------------------------------------------------
for col in X.columns:
    if X[col].dtype == "object":
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# -------------------------------------------------
# Apply preprocessing pipeline (fitted on TRAIN data)
# -------------------------------------------------
preprocess_pipeline = joblib.load("model/preprocess_pipeline.pkl")
X = preprocess_pipeline.transform(X)

# -------------------------------------------------
# Load trained model
# -------------------------------------------------
model = joblib.load(f"model/{model_name.replace(' ', '_')}.pkl")

# -------------------------------------------------
# Predictions
# -------------------------------------------------
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# -------------------------------------------------
# Metrics as cards
# -------------------------------------------------
st.subheader("Evaluation Metrics")

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

c1.metric("Accuracy", f"{accuracy_score(y, y_pred):.4f}")
c2.metric("AUC", f"{roc_auc_score(y, y_prob):.4f}")
c3.metric("Precision", f"{precision_score(y, y_pred):.4f}")
c4.metric("Recall", f"{recall_score(y, y_pred):.4f}")
c5.metric("F1 Score", f"{f1_score(y, y_pred):.4f}")
c6.metric("MCC", f"{matthews_corrcoef(y, y_pred):.4f}")

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# -------------------------------------------------
# Classification Report
# -------------------------------------------------
st.subheader("Classification Report")

report_df = pd.DataFrame(
    classification_report(y, y_pred, output_dict=True)
).transpose()

st.dataframe(report_df.style.background_gradient(cmap="Greens"))
