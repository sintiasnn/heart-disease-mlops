# Heart Disease MLOps Pipeline

![CI](https://github.com/sintiasnn/heart-disease-mlops/actions/workflows/ci.yml/badge.svg)
[![DagsHub](https://img.shields.io/badge/DagsHub-Experiments-orange?logo=dagshub)](https://dagshub.com/sintiasnn/heart-disease-mlops.mlflow)
[![Docker Hub](https://img.shields.io/badge/Docker%20Hub-Image-blue?logo=docker)](https://hub.docker.com/r/sintiasnn/heart-disease-mlops)
![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-2.19.0-blue?logo=mlflow)

An end-to-end MLOps pipeline for heart disease prediction using the Heart Disease dataset from the UCI Machine Learning Repository. This project covers the full MLOps lifecycle — from experimentation and model building to CI/CD workflow automation and real-time monitoring.

---

## Table of Contents
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Experiment Results](#experiment-results)
- [Monitoring](#monitoring)
- [Author](#author)

---

## Overview

This project builds a complete MLOps pipeline to predict whether a patient has heart disease (binary classification) based on clinical data.

**Pipeline stages:**
1. **Experimentation** — EDA and automated data preprocessing
2. **Modelling** — Model training with MLflow experiment tracking, logged to DagsHub
3. **CI/CD** — Automated retraining with GitHub Actions, artifact storage, and Docker Hub deployment
4. **Monitoring** — Real-time monitoring with Prometheus & Grafana

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML Framework | Scikit-Learn |
| Experiment Tracking | MLflow 2.19.0 + DagsHub |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker + Docker Hub |
| Data Source | UCI ML Repository |

---

## Project Structure

```
heart-disease-mlops/
├── README.md
├── experimentation/
│   ├── heart_disease_raw.csv
│   └── preprocessing/
│       ├── experiment.ipynb
│       ├── automate.py
│       └── heart_disease_preprocessing.csv
├── modelling/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── requirements.txt
│   ├── .env.example
│   └── .env              # (not committed)
├── workflow-ci/
│   ├── .github/
│   │   └── workflows/
│   │       └── ci.yml
│   └── MLProject/
│       ├── modelling.py
│       ├── conda.yaml
│       └── MLProject
└── monitoring/
    ├── docker-compose.yml
    ├── prometheus.yml
    ├── prometheus_exporter.py
    ├── inference.py
    ├── .env.example
    └── grafana/
        └── provisioning/
            ├── datasources/
            │   └── prometheus.yml
            └── alerting/
                └── alerts.yml
```

---

## Dataset

- **Source**: [UCI ML Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Size**: 303 rows, 13 features + 1 target
- **Task**: Binary Classification (0 = no disease, 1 = heart disease)
- **Features**: Mix of numerical and categorical (age, sex, cp, trestbps, chol, etc.)

---

## Getting Started

### Prerequisites
- Python 3.12
- Docker
- Git

### 1. Clone Repository
```bash
git clone https://github.com/sintiasnn/heart-disease-mlops.git
cd heart-disease-mlops
```

### 2. Setup Environment
```bash
python3.12 -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r modelling/requirements.txt
```

### 3. Configure Environment Variables
```bash
cp modelling/.env.example modelling/.env
# Edit modelling/.env and fill in your credentials
```

Required variables in `modelling/.env`:
```
MLFLOW_TRACKING_URI=https://dagshub.com/<your_username>/heart-disease-mlops.mlflow
MLFLOW_TRACKING_USERNAME=<your_dagshub_username>
MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

### 4. Run Automated Preprocessing
```bash
cd experimentation/preprocessing
python automate.py
cp heart_disease_preprocessing_train.csv ../../modelling/
cp heart_disease_preprocessing_test.csv ../../modelling/
```

### 5. Train Model
```bash
cd modelling

# Baseline (autolog)
python modelling.py

# With hyperparameter tuning (manual logging)
python modelling_tuning.py
```

Experiments and artifacts (including `confusion_matrix.png` and `feature_importance.csv`) are logged to [DagsHub](https://dagshub.com/sintiasnn/heart-disease-mlops).

### 6. Open MLflow UI (via DagsHub)
```
https://dagshub.com/sintiasnn/heart-disease-mlops.mlflow
```

Or locally:
```bash
mlflow ui
# Open http://127.0.0.1:5000
```

### 7. Serve Model
```bash
mlflow models serve \
  -m mlruns/<experiment_id>/<run_id>/artifacts/model \
  -p 5001 \
  --env-manager=local \
  --no-conda
```

### 8. Run Monitoring Stack
```bash
# Terminal 1 - Prometheus Exporter
cd monitoring
python prometheus_exporter.py

# Terminal 2 - Prometheus + Grafana via Docker Compose
cd monitoring
cp .env.example .env
# Edit .env and fill in your SMTP credentials (optional)
docker compose up -d

# Open http://localhost:3000 (default: admin/admin)
```

> **Grafana Provisioning**: Datasource (Prometheus) dan alert rules sudah otomatis ter-konfigurasi via file di `grafana/provisioning/`. Tidak perlu setup manual di UI.

> **Gmail App Password**: To use Gmail SMTP, enable 2-Step Verification on your Google account, then generate an App Password at [myaccount.google.com](https://myaccount.google.com) → Security → App passwords.

### 9. Run Inference
```bash
cd monitoring
python inference.py
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs automatically on every push to `main`:

1. Generates preprocessed dataset via `automate.py`
2. Runs MLflow Project and logs experiments to DagsHub
3. Uploads `mlruns/`, `confusion_matrix.png`, and `feature_importance.csv` as GitHub Artifacts
4. Builds a Docker image using `mlflow models build-docker`
5. Pushes the image to Docker Hub

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `MLFLOW_TRACKING_URI` | DagsHub MLflow tracking URI |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub access token |
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

---

## Experiment Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Random Forest (Baseline) | 0.7368 | 0.7273 | 0.6400 | 0.6809 | — |
| Random Forest (Tuning) | **0.7719** | **0.8000** | 0.6400 | **0.7111** | **0.8575** |

**Best Parameters (GridSearchCV):**
- `n_estimators`: 100
- `max_depth`: 5
- `min_samples_split`: 2

---

## Monitoring

Real-time monitoring using Prometheus and Grafana with the following metrics:

| Metric | Description |
|---|---|
| `prediction_requests_total` | Total number of prediction requests |
| `prediction_class_total` | Predictions per class (sick/not sick) |
| `prediction_latency_seconds` | Prediction request latency |
| `model_status` | Model serving status (1=up, 0=down) |
| `prediction_positive_total` | Total positive predictions |
| `prediction_negative_total` | Total negative predictions |
| `model_accuracy` | Model accuracy |
| `prediction_error_total` | Total failed prediction requests |
| `prediction_positive_rate` | Ratio of positive predictions over total |
| `model_response_size_bytes` | Size of the last model response in bytes |

**Alert Rules (auto-provisioned):**

| Alert | Condition | Severity |
|---|---|---|
| Model Down | `model_status == 0` for > 1 min | Critical |
| High Error Rate | `rate(prediction_error_total[5m]) > 0.1` for > 2 min | Warning |
| High Latency | P95 latency > 1s for > 2 min | Warning |

---

## Author

**Ni Putu Sintia Wati**
- GitHub: [@sintiasnn](https://github.com/sintiasnn)

---

## License

This project is open source and available under the [MIT License](LICENSE).
