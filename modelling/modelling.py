import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
# MLflow Autolog
# ========================
mlflow.set_experiment("heart-disease-classification")

with mlflow.start_run(run_name="RandomForest-Baseline"):
    # Aktifkan autolog
    mlflow.sklearn.autolog()

    print(">>> Melatih model...")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluasi
    y_pred = model.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)

    print(">>> Hasil Evaluasi:")
    print(f"    Accuracy  : {accuracy:.4f}")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1 Score  : {f1:.4f}")

print("\n>>> Training selesai! Buka MLflow UI dengan:")
print("    mlflow ui")