# =========================================
# Insurance Claims Prediction Dashboard (FINAL ERROR-FREE VERSION)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix

# -----------------------
# Streamlit Layout
# -----------------------
st.title("🚗 Insurance Claims Prediction & Visualization Dashboard (Final Version)")
st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #0043ce, #F4F6F9, #75cfff); }
section[data-testid="stSidebar"] { background: #28A745 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# 1. Upload CSV
# -----------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Dataset Loaded Successfully! Shape: {df.shape}")

    # -----------------------
    # 2. Preprocessing
    # -----------------------
    expected_numeric = ["CREDIT_SCORE", "CHILDREN", "ANNUAL_MILEAGE",
                        "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"]

    expected_categorical = ["AGE", "GENDER", "RACE", "DRIVING_EXPERIENCE", "EDUCATION", 
                            "INCOME", "VEHICLE_OWNERSHIP", "VEHICLE_YEAR", 
                            "MARRIED", "VEHICLE_TYPE"]

    # Fill numeric columns
    for col in expected_numeric:
        df[col] = df[col].fillna(df[col].median()) if col in df else 0

    # Fill categorical columns
    for col in expected_categorical:
        df[col] = df[col].fillna("Unknown") if col in df else "Unknown"

    # Label Encoding
    categorical_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in categorical_cols:
        try:
            df[col] = le.fit_transform(df[col])
        except:
            df[col] = le.fit_transform(df[col].astype(str))

    # Identify features and target
    if "OUTCOME" in df.columns:
        X = df.drop("OUTCOME", axis=1)
        y = df["OUTCOME"]
    else:
        X = df.copy()
        y = None

    # -----------------------
    # 3. Train Model
    # -----------------------
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    svm = SVC(probability=True)
    log_reg = LogisticRegression(max_iter=1000)

    ensemble_model = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("lr", log_reg)],
        voting="soft"
    )

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        ensemble_model.fit(X_train, y_train)
    else:
        ensemble_model.fit(X, np.zeros(X.shape[0]))

    # -----------------------
    # 4. Predictions
    # -----------------------
    df["Claim_Prob"] = ensemble_model.predict_proba(X)[:, 1]

    threshold = st.slider(
        "Select Risk Threshold (%)", 50, 100, 80
    ) / 100

    df["Risk_Flag"] = df["Claim_Prob"].apply(
        lambda x: "High Risk" if x >= threshold else "Low Risk"
    )

    st.header("📋 Prediction Table")
    st.dataframe(df)

    st.download_button(
        "Download Predictions",
        df.to_csv(index=False),
        "Predictions.csv"
    )

    # -----------------------
    # 5. Demographic Visualizations (Objective v)
    # -----------------------
    st.header("📊 Policyholder Demographic Visualizations")

    demographic_columns = ["AGE", "GENDER", "EDUCATION", "INCOME", "RACE", 
                           "VEHICLE_TYPE", "DRIVING_EXPERIENCE", "VEHICLE_YEAR"]

    selected_demo = st.selectbox("Select a demographic variable:", demographic_columns)

    fig_demo = px.box(
        df, x=selected_demo, y="Claim_Prob", 
        color="Risk_Flag",
        title=f"Claim Probability by {selected_demo}"
    )
    st.plotly_chart(fig_demo, use_container_width=True)

    # -----------------------
    # 6. Fraud Scatter Plot
    # -----------------------
    st.header("🚨 Fraud Detection Scatter Plot")

    fig_fraud = px.scatter(
        df,
        x="Claim_Prob",
        y="ANNUAL_MILEAGE",
        color="Risk_Flag",
        hover_data=df.columns,
        title="Fraud Risk Distribution"
    )
    st.plotly_chart(fig_fraud, use_container_width=True)

    # -----------------------
    # 7. Correlation Matrix (Objective iv) — FIXED
    # -----------------------
    st.header("🔎 Correlation Matrix (Numeric Only)")

    numeric_df = df.select_dtypes(include=["number"])  # FIXED

    if numeric_df.shape[1] > 1:
        corr = numeric_df.corr()

        fig_corr, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(corr, cmap="coolwarm", ax=ax)
        st.pyplot(fig_corr)
    else:
        st.warning("Not enough numeric columns for correlation analysis.")

    # -----------------------
    # 8. Feature Importance (Objective iv) — FIXED
    # -----------------------
    st.header("📌 Random Forest Feature Importance")

    numeric_X = X.select_dtypes(include=["number"])  # FIXED
    rf.fit(numeric_X, y)

    importances = pd.DataFrame({
        "Feature": numeric_X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importances.head(10))

    fig_imp = px.bar(
        importances.head(10),
        x="Feature",
        y="Importance",
        title="Top 10 Most Important Features"
    )
    st.plotly_chart(fig_imp)

    # -----------------------
    # 9. Model Performance
    # -----------------------
    if y is not None:
        st.header("📈 Model Evaluation")

        y_pred = ensemble_model.predict(X_test)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig_cm)

        # ROC curve
        y_probs = ensemble_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        fig_roc = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(fig_roc)

    # -----------------------
    # 10. High-Risk Customers
    # -----------------------
    st.header("🚨 Top 10 High-Risk Policyholders")

    top_risk = df.sort_values("Claim_Prob", ascending=False).head(10)
    st.dataframe(top_risk)

    st.download_button(
        "Download High-Risk Customers",
        top_risk.to_csv(index=False),
        "HighRisk.csv"
    )

    # -----------------------
    # 11. Lookup System
    # -----------------------
    st.header("🔍 Policyholder Lookup")

    df["_ID_"] = df["POSTAL_CODE"] if "POSTAL_CODE" in df else df.index

    search_id = st.text_input("Enter Postal Code or Index:")

    if search_id:
        try:
            search_id = int(search_id)
            result = df[df["_ID_"] == search_id]
            if len(result) > 0:
                st.json(result.iloc[0].to_dict())
            else:
                st.warning("No matching policyholder found.")
        except:
            st.error("Please enter a valid numeric ID.")
