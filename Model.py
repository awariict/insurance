import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# -----------------------------
# Step 1: Load dataset
# -----------------------------
df = pd.read_csv("insurance_data.csv")

# -----------------------------
# Step 2: Preprocessing
# -----------------------------
categorical_cols = ["AGE", "GENDER", "RACE", "DRIVING_EXPERIENCE",
                    "EDUCATION", "INCOME", "VEHICLE_YEAR", "VEHICLE_TYPE"]

# Encode categorical values
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Fix missing values
df["CHILDREN"] = df["CHILDREN"].fillna(df["CHILDREN"].median())
df["ANNUAL_MILEAGE"] = df["ANNUAL_MILEAGE"].fillna(df["ANNUAL_MILEAGE"].median())

# Features & target
X = df.drop(columns=["ID", "OUTCOME"])
y = df["OUTCOME"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# Step 3: Train Models
# -----------------------------
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
svm = SVC(kernel="rbf", probability=True, random_state=42)

models = {
    "Random Forest": rf,
    "Logistic Regression": lr,
    "SVM": svm
}

# Fit models
for name, model in models.items():
    model.fit(X_train, y_train)

# -----------------------------
# Step 4: Evaluation & Reports
# -----------------------------
for name, model in models.items():
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# Step 5: Confusion Matrices
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, model) in zip(axes, models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# -----------------------------
# Step 6: ROC Curves
# -----------------------------
plt.figure(figsize=(8, 6))
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.show()

# -----------------------------
# Step 7: Feature Importance (Random Forest)
# -----------------------------
importances = rf.feature_importances_
feature_names = df.drop(columns=["ID", "OUTCOME"]).columns
fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=fi_df)
plt.title("Random Forest Feature Importance")
plt.show()

# -----------------------------
# Step 8: Hyperparameter Tuning
# -----------------------------
# Random Forest
rf_params = {"n_estimators": [50, 100, 200],
             "max_depth": [None, 5, 10],
             "min_samples_split": [2, 5]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring="f1")
rf_grid.fit(X_train, y_train)
print("\nBest Random Forest Params:", rf_grid.best_params_)

# Logistic Regression
lr_params = {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}
lr_grid = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_params, cv=3, scoring="f1")
lr_grid.fit(X_train, y_train)
print("\nBest Logistic Regression Params:", lr_grid.best_params_)

# SVM
svm_params = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42), svm_params, cv=3, scoring="f1")
svm_grid.fit(X_train, y_train)
print("\nBest SVM Params:", svm_grid.best_params_)

# -----------------------------
# Step 9: Compare Before vs After Tuning
# -----------------------------
def evaluate_model(name, model, X_test, y_test):
    """Return evaluation metrics for a model."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = model.decision_function(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

# Evaluate before tuning
results_before = [
    evaluate_model("Random Forest (Default)", rf, X_test, y_test),
    evaluate_model("Logistic Regression (Default)", lr, X_test, y_test),
    evaluate_model("SVM (Default)", svm, X_test, y_test),
]

# Evaluate after tuning (best estimators)
results_after = [
    evaluate_model("Random Forest (Tuned)", rf_grid.best_estimator_, X_test, y_test),
    evaluate_model("Logistic Regression (Tuned)", lr_grid.best_estimator_, X_test, y_test),
    evaluate_model("SVM (Tuned)", svm_grid.best_estimator_, X_test, y_test),
]

# Combine into one DataFrame
final_results = pd.DataFrame(results_before + results_after)
print("\n=== Model Performance Before vs After Tuning ===\n")
print(final_results)
