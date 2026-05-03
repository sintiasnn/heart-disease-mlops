import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')

# ========================
# Load Dataset
# ========================
print(">>> Memuat dataset...")
train_df = pd.read_csv('heart_disease_preprocessing_train.csv')
test_df = pd.read_csv('heart_disease_preprocessing_test.csv')

X_train = train_df.drop('target', axis=1)
y_train = train_df['target']
X_test = test_df.drop('target', axis=1)
y_test = test_df['target']

print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")

# ========================
# Hyperparameter Tuning
# ========================
print(">>> Melakukan hyperparameter tuning...")

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5],
}

base_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"    Best params: {best_params}")

# ========================
# MLflow Manual Logging
# ========================
mlflow.set_experiment("heart-disease-classification")

with mlflow.start_run(run_name="RandomForest-Tuning"):

    # Train model dengan best params
    best_model = RandomForestClassifier(
        **best_params,
        random_state=42
    )
    best_model.fit(X_train, y_train)

    # Prediksi
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # Hitung metrik
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # Log parameters secara manual
    mlflow.log_param("n_estimators", best_params['n_estimators'])
    mlflow.log_param("max_depth", best_params['max_depth'])
    mlflow.log_param("min_samples_split", best_params['min_samples_split'])
    mlflow.log_param("random_state", 42)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "f1")

    # Log metrics secara manual
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    # Print hasil
    print("\n>>> Hasil Evaluasi:")
    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")
    print(f"    ROC AUC   : {roc_auc:.4f}")
    print(f"    Best CV F1: {grid_search.best_score_:.4f}")

    print("\n>>> Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\n>>> Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

print("\n>>> Training selesai! Buka MLflow UI dengan:")
print("    mlflow ui")