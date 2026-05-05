import os
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
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
# Hyperparameter Tuning
# ========================
print(">>> Performing hyperparameter tuning...")

param_grid = {
    'n_estimators':    [50, 100, 200],
    'max_depth':       [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5, scoring='f1', n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"    Best params: {best_params}")

# ========================
# MLflow Manual Logging
# ========================
mlflow.set_experiment("heart-disease-classification")

with mlflow.start_run(run_name="RandomForest-Tuning"):

    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # Log params
    mlflow.log_param("n_estimators",      best_params['n_estimators'])
    mlflow.log_param("max_depth",         best_params['max_depth'])
    mlflow.log_param("min_samples_split", best_params['min_samples_split'])
    mlflow.log_param("random_state", 42)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "f1")

    # Log metrics
    mlflow.log_metric("accuracy",      accuracy)
    mlflow.log_metric("precision",     precision)
    mlflow.log_metric("recall",        recall)
    mlflow.log_metric("f1_score",      f1)
    mlflow.log_metric("roc_auc",       roc_auc)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    # --- Artefak 1: Confusion Matrix ---
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    fig.savefig("confusion_matrix.png")
    plt.close(fig)
    mlflow.log_artifact("confusion_matrix.png")

    # --- Artefak 2: Feature Importance ---
    feat_imp = pd.DataFrame({
        'feature':    X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    feat_imp.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    print(f"\n>>> Evaluation Results:")
    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC AUC   : {roc_auc:.4f}")
    print(f"\n>>> Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f">>> Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

print("\n>>> Training finished!")
