# =========================================
# Insurance Claims Prediction Dashboard
# =========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# -----------------------
# Step 1: Upload Dataset
# -----------------------
st.title("🚗 Insurance Claims Prediction Dashboard")
st.markdown("Upload a CSV file with policyholder data. The system predicts claim probabilities and flags high-risk users for fraud detection and decision-making.")

uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -----------------------
    # Step 2: Preprocessing
    # -----------------------
    df.fillna(df.median(numeric_only=True), inplace=True)  # Fill numeric missing values
    categorical_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Features & Target
    X = df.drop("OUTCOME", axis=1)
    y = df["OUTCOME"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # -----------------------
    # Step 3: Train Soft-Voting Ensemble
    # -----------------------
    rf = RandomForestClassifier(random_state=42)
    log_reg = LogisticRegression(max_iter=1000)
    svm = SVC(probability=True)

    ensemble_model = VotingClassifier(
        estimators=[("rf", rf), ("lr", log_reg), ("svm", svm)],
        voting="soft"
    )
    ensemble_model.fit(X_train, y_train)

    # -----------------------
    # Step 4: Predict Claim Probabilities
    # -----------------------
    df["Claim_Prob"] = ensemble_model.predict_proba(X)[:, 1]
    df["Risk_Flag"] = df["Claim_Prob"].apply(lambda x: "High Risk" if x >= 0.8 else "Low Risk")

    # -----------------------
    # Step 5: Dynamic Advice
    # -----------------------
    def generate_advice(row):
        if row["Risk_Flag"] == "High Risk":
            return "Insurer: Investigate; Claimant: Provide full documentation"
        else:
            return "Insurer: Standard review; Claimant: Proceed normally"

    df["Dynamic_Advice"] = df.apply(generate_advice, axis=1)
    df["Estimated_Claim_Cost"] = df["Claim_Prob"] * 100000  # Example cost multiplier
    total_liability = df["Estimated_Claim_Cost"].sum()

    # ===================== TAB LAYOUT =====================
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview",
        "🚨 Fraud Detection",
        "📈 Risk & Advice"
    ])

    # ===================== TAB 1: OVERVIEW =====================
    with tab1:
        st.header("📊 Group-Level Risk Summary")
        group_columns = ["INCOME", "VEHICLE_TYPE", "DUIS", "PAST_ACCIDENTS"]
        available_groups = [col for col in group_columns if col in df.columns]

        if available_groups:
            group_choice = st.selectbox("Group policyholders by:", available_groups)
            group_summary = df.groupby(group_choice)["Claim_Prob"].mean().reset_index()
            st.bar_chart(group_summary.set_index(group_choice))

        st.subheader("🏆 Top High-Risk Policyholders per Group")
        top_group_risk = df.sort_values(["Claim_Prob"], ascending=False).groupby(group_choice).head(10)
        st.dataframe(top_group_risk[["Claim_Prob", "Risk_Flag", "Dynamic_Advice", "Estimated_Claim_Cost"]].reset_index(drop=True))
        st.download_button(
            label="📥 Download Top High-Risk Policyholders",
            data=top_group_risk.to_csv(index=False),
            file_name=f"top_high_risk_{group_choice}.csv",
            mime="text/csv"
        )

    # ===================== TAB 2: FRAUD DETECTION =====================
    with tab2:
        st.header("🚨 Fraud Detection Visualizations")

        # Scatterplot
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

        # Risk Heatmap
        st.subheader("📍 Risk Heatmap")
        x_col = "ANNUAL_MILEAGE" if "ANNUAL_MILEAGE" in df.columns else df.columns[0]
        y_col = "PAST_ACCIDENTS" if "PAST_ACCIDENTS" in df.columns else df.columns[1]
        heatmap_data = df.pivot_table(index=y_col, columns=x_col, values="Claim_Prob", aggfunc="mean", fill_value=0)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(heatmap_data, cmap="Reds", ax=ax)
        ax.set_title("High-Risk Policyholders Heatmap")
        st.pyplot(fig)

    # ===================== TAB 3: RISK & ADVICE =====================
    with tab3:
        st.header("📈 Policyholder Risk & Advice")
        st.metric("Total Estimated Liability", f"₦{total_liability:,.0f}")

        st.download_button(
            label="📥 Download Full Predicted Results",
            data=df.to_csv(index=False),
            file_name="predicted_policyholders.csv",
            mime="text/csv"
        )

        st.subheader("Full Policyholder Predictions")
        st.dataframe(df)
