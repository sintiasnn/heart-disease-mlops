import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

# ========================
# DagsHub MLflow Tracking
# ========================
load_dotenv()

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# ========================
# Load Dataset
# ========================
print(">>> Loading dataset...")
train_df = pd.read_csv('heart_disease_preprocessing_train.csv')
test_df  = pd.read_csv('heart_disease_preprocessing_test.csv')

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test  = test_df.drop('target', axis=1)
y_test  = test_df['target']

print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")

# ========================
# MLflow Autolog
# ========================
mlflow.set_experiment("heart-disease-classification")

with mlflow.start_run(run_name="RandomForest-Baseline"):
    mlflow.sklearn.autolog()

    print(">>> Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")

    # --- Artefak 1: Confusion Matrix ---
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig("confusion_matrix.png")
    plt.close(fig)
    mlflow.log_artifact("confusion_matrix.png")

    # --- Artefak 2: Feature Importance ---
    feat_imp = pd.DataFrame({
        'feature':   X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    feat_imp.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

print("\n>>> Training finished!")
