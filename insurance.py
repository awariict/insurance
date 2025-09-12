# =========================================
# Insurance Claims Prediction Dashboard
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
from sklearn.calibration import CalibratedClassifierCV

# -----------------------
# Streamlit Layout
# -----------------------
st.title("🚗 Insurance Claims Prediction Dashboard")
st.markdown("""
<style>
.stApp { background: linear-gradient(to right, #0043ce, #F4F6F9, #75cfff); }
section[data-testid="stSidebar"] { background: #28A745 !important; }
.sidebar-content { display: flex; flex-direction: column; align-items: center; gap: 12px; margin-top: 20px; }
.sidebar-btn { width: 180px; height: 44px; background-color: white !important; color: #28A745 !important; font-weight: bold; border-radius: 8px; border: none; margin: 0 auto; display: block; }
.big-font { font-size:20px !important; }
.card { background: white; color: black !important; padding: 12px; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)
# -----------------------
# Step 1: Upload CSV
# -----------------------
uploaded_file = st.file_uploader("Upload your policyholder CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # -----------------------
    # Step 2: Preprocessing
    # -----------------------
    expected_numeric = ["CREDIT_SCORE", "CHILDREN", "ANNUAL_MILEAGE",
                        "SPEEDING_VIOLATIONS", "DUIS", "PAST_ACCIDENTS"]
    expected_categorical = ["AGE", "GENDER", "RACE", "DRIVING_EXPERIENCE", "EDUCATION", 
                            "INCOME", "VEHICLE_OWNERSHIP", "VEHICLE_YEAR", "MARRIED", "VEHICLE_TYPE"]

    # Fill numeric columns
    for col in expected_numeric:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Fill categorical columns
    for col in expected_categorical:
        if col not in df.columns:
            df[col] = "Unknown"
        else:
            df[col].fillna("Unknown", inplace=True)

    # Encode categorical features
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
    # Step 3: Train Ensemble Model
    # -----------------------
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    log_reg = LogisticRegression(max_iter=1000)
    svm = SVC(probability=True)

    ensemble_model = VotingClassifier(
        estimators=[("rf", rf), ("lr", log_reg), ("svm", svm)],
        voting="soft"
    )

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        ensemble_model.fit(X_train, y_train)
    else:
        ensemble_model.fit(X, np.zeros(X.shape[0]))

    # -----------------------
    # Step 4: Predictions and Risk Assessment
    # -----------------------
    df["Claim_Prob"] = ensemble_model.predict_proba(X)[:, 1]
    threshold = st.slider("Set high-risk threshold (%)", 50, 100, 80) / 100
    df["Risk_Flag"] = df["Claim_Prob"].apply(lambda x: "High Risk" if x >= threshold else "Low Risk")
    df["Advice_to_Claimant"] = df["Risk_Flag"].apply(lambda x: "Monitor driving habits" if x=="Low Risk" else "Be prepared for review")
    df["Advice_to_Insurer"] = df["Risk_Flag"].apply(lambda x: "Fast-track claim" if x=="Low Risk" else "Investigate for fraud")

    # -----------------------
    # Step 5: Display Predicted Results
    # -----------------------
    st.header("📋 Predicted Results")
    st.dataframe(df)
    st.download_button(
        label="📥 Download Predictions & Advice",
        data=df.to_csv(index=False),
        file_name="insurance_predictions.csv",
        mime="text/csv"
    )

    # -----------------------
    # Step 6: Interactive Visualizations
    # -----------------------
    st.header("📊 Group-Level Risk Summary")
    group_columns = ["INCOME", "VEHICLE_TYPE", "DUIS", "PAST_ACCIDENTS"]
    available_groups = [col for col in group_columns if col in df.columns]
    if available_groups:
        group_choice = st.selectbox("Group policyholders by:", available_groups)
        group_summary = df.groupby(group_choice)["Claim_Prob"].mean().reset_index()
        st.bar_chart(group_summary.set_index(group_choice))
    else:
        st.warning("⚠️ No available columns for group-level summary.")

    st.header("🚨 Fraud Detection Scatterplot")
    y_axis_col = "ANNUAL_MILEAGE" if "ANNUAL_MILEAGE" in df.columns else df.columns[0]
    fig = px.scatter(
        df,
        x="Claim_Prob",
        y=y_axis_col,
        color="Risk_Flag",
        hover_data=list(df.columns),
        title="Policyholder Risk Distribution",
        color_discrete_map={"High Risk": "red", "Low Risk": "blue"}
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------
    # Step 7: Model Performance (if labels exist)
    # -----------------------
    if y is not None:
        st.header("📈 Model Performance")
        y_pred = ensemble_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        # ROC Curve
        y_probs = ensemble_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)
        st.subheader(f"ROC Curve (AUC={roc_auc:.2f})")
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        st.pyplot(plt)

    # -----------------------
    # Step 8: Top 10 High-Risk Policyholders
    # -----------------------
    st.header("🚨 Top 10 High-Risk Policyholders")
    top_risk = df.sort_values(by="Claim_Prob", ascending=False).head(10)
    st.dataframe(top_risk[["Claim_Prob", "Risk_Flag", "Advice_to_Claimant", "Advice_to_Insurer"]])
    st.download_button(
        label="📥 Download Top 10 High-Risk Policyholders",
        data=top_risk.to_csv(index=False),
        file_name="top10_high_risk_policyholders.csv",
        mime="text/csv"
    )

    # -----------------------
    # Step 9: Visual Alerts for Fraud & Fast-Track
    # -----------------------
    st.header("🚨 Fraud & Fast-Track Alerts")
    high_risk = df[df["Risk_Flag"] == "High Risk"]
    low_risk = df[df["Risk_Flag"] == "Low Risk"]

    st.subheader("🔴 High-Risk / Potential Fraud")
    if not high_risk.empty:
        st.dataframe(high_risk[["Claim_Prob", "Advice_to_Claimant", "Advice_to_Insurer"]])
        st.markdown("<p style='color:red'>⚠️ Investigate these policyholders!</p>", unsafe_allow_html=True)
    else:
        st.success("No high-risk policyholders detected.")

    st.subheader("🟢 Low-Risk / Fast-Track")
    if not low_risk.empty:
        st.dataframe(low_risk[["Claim_Prob", "Advice_to_Claimant", "Advice_to_Insurer"]])
        st.markdown("<p style='color:green'>✅ Fast-track these claims!</p>", unsafe_allow_html=True)
    else:
        st.warning("No low-risk policyholders available for fast-track.")

    # -----------------------
    # Step 10: Policyholder Lookup
    # -----------------------
    st.header("🔍 Policyholder Lookup")
    id_column = "POSTAL_CODE" if "POSTAL_CODE" in df.columns else df.index.name
    df["_id_"] = df[id_column] if id_column else df.index

    search_id = st.text_input("Enter Policyholder Postal Code:")

    if search_id:
        try:
            search_id_int = int(search_id)
            if search_id_int in df["_id_"].values:
                person = df[df["_id_"] == search_id_int].iloc[0]
                st.subheader(f"Policyholder: {search_id_int}")
                st.metric("Predicted Claim Probability", f"{person['Claim_Prob']:.2%}")
                st.metric("Risk Flag", person["Risk_Flag"])
                st.metric("Advice to Claimant", person["Advice_to_Claimant"])
                st.metric("Advice to Insurer", person["Advice_to_Insurer"])
                st.write("**Full Policyholder Details:**")
                st.json(person.to_dict())
            else:
                st.warning("⚠️ Policyholder ID not found in dataset.")
        except:
            st.error("❌ Enter a valid numeric Policyholder ID.")





