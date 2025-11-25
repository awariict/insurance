# =========================================
# Insurance Claims Prediction Dashboard (Upgraded)
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix

# -----------------------
# Streamlit Layout
# -----------------------
st.title("🚗 Insurance Claims Prediction & Visualization Dashboard (Enhanced Version)")
st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #0043ce, #F4F6F9, #75cfff); }
section[data-testid="stSidebar"] { background: #28A745 !important; }
</style>
""", unsafe_allow_html=True)

# -----------------------
# 1. Upload CSV
# -----------------------
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

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

    for col in expected_numeric:
        df[col] = df[col].fillna(df[col].median()) if col in df else 0

    for col in expected_categorical:
        df[col] = df[col].fillna("Unknown") if col in df else "Unknown"

    # Label Encoding
    categorical_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Separate features and target
    if "OUTCOME" in df.columns:
        X = df.drop("OUTCOME", axis=1)
        y = df["OUTCOME"]
    else:
        X = df.copy()
        y = None

    # -----------------------
    # 3. Train Ensemble Model
    # -----------------------
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    svm = SVC(probability=True)
    log_reg = LogisticRegression(max_iter=1000)

    model = VotingClassifier(
        estimators=[("rf", rf), ("svm", svm), ("lr", log_reg)],
        voting="soft"
    )

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
    else:
        model.fit(X, np.zeros(X.shape[0]))

    # -----------------------
    # 4. Predictions
    # -----------------------
    df["Claim_Prob"] = model.predict_proba(X)[:, 1]
    threshold = st.slider("Select Risk Threshold (%)", 50, 100, 80) / 100
    df["Risk_Flag"] = df["Claim_Prob"].apply(lambda x: "High Risk" if x >= threshold else "Low Risk")

    # -----------------------
    # 5. Show Results
    # -----------------------
    st.header("📋 Prediction Output Table")
    st.dataframe(df)

    st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv")

    # -----------------------
    # 6. GROUP VISUALIZATION — Objective (v)
    # -----------------------
    st.header("📊 Policyholder Demographic Visualizations")

    demographic_columns = ["AGE", "GENDER", "EDUCATION", "INCOME", "RACE", 
                           "VEHICLE_TYPE", "DRIVING_EXPERIENCE", "VEHICLE_YEAR"]

    selected_demo = st.selectbox("Select a demographic variable:", demographic_columns)

    fig_demo = px.box(df, x=selected_demo, y="Claim_Prob", color="Risk_Flag",
                      title=f"Claim Probability by {selected_demo}")
    st.plotly_chart(fig_demo, use_container_width=True)

    # -----------------------
    # 7. FRAUD SCATTERPLOT
    # -----------------------
    st.header("🚨 Fraud Detection Scatter Plot")
    fig_fraud = px.scatter(
        df, x="Claim_Prob", y="ANNUAL_MILEAGE",
        color="Risk_Flag",
        title="Fraud Risk Analysis",
        hover_data=df.columns
    )
    st.plotly_chart(fig_fraud, use_container_width=True)

    # -----------------------
    # 8. CORRELATION MATRIX — Objective (iv)
    # -----------------------
    st.header("🔎 Correlation Matrix (Key Risk Feature Aggregation)")

    corr = df.corr()
    fig_corr, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

    # -----------------------
    # 9. FEATURE IMPORTANCE — Objective (iv)
    # -----------------------
    st.header("📌 Random Forest Feature Importance (Top Predictors)")

    rf.fit(X, y)
    importances = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(importances.head(10))

    fig_imp = px.bar(importances.head(10), x="Feature", y="Importance",
                     title="Top 10 Most Important Risk Factors")
    st.plotly_chart(fig_imp)

    # -----------------------
    # 10. SHAP VALUES (Explanation)
    # -----------------------
    st.header("🧠 Explainability: SHAP Feature Impact (Optional)")

    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)
        st.write("SHAP Summary Plot:")

        fig_shap = plt.figure()
        shap.summary_plot(shap_values[1], X)
        st.pyplot(fig_shap)

    except Exception:
        st.warning("SHAP could not be generated due to system limitations.")

    # -----------------------
    # 11. MODEL PERFORMANCE
    # -----------------------
    if y is not None:
        st.header("📈 Model Evaluation Metrics")

        y_pred = model.predict(X_test)

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig_cm)

        # ROC Curve
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        fig_roc = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        st.pyplot(fig_roc)

    # -----------------------
    # 12. HIGH-RISK LIST
    # -----------------------
    st.header("🚨 Top 10 High-Risk Policyholders")
    top_risk = df.sort_values("Claim_Prob", ascending=False).head(10)
    st.dataframe(top_risk)

    st.download_button("Download High-Risk List", top_risk.to_csv(index=False),
                       "HighRiskPolicyholders.csv")

    # -----------------------
    # 13. LOOKUP SYSTEM
    # -----------------------
    st.header("🔍 Policyholder Lookup Tool")

    if "POSTAL_CODE" in df.columns:
        df["_ID_"] = df["POSTAL_CODE"]
    else:
        df["_ID_"] = df.index

    search = st.text_input("Enter Postal Code or Index:")

    if search:
        try:
            search = int(search)
            person = df[df["_ID_"] == search]
            if len(person) > 0:
                st.json(person.iloc[0].to_dict())
            else:
                st.warning("No matching policyholder found.")
        except:
            st.error("Enter a valid number.")
