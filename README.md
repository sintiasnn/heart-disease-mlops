# Heart Disease MLOps Pipeline

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
2. **Modelling** — Model training with MLflow experiment tracking
3. **CI/CD** — Automated retraining with GitHub Actions and MLflow Projects
4. **Monitoring** — Real-time monitoring with Prometheus & Grafana

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| ML Framework | Scikit-Learn |
| Experiment Tracking | MLflow 2.19.0 |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker |
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
│   └── requirements.txt
├── workflow-ci/
│   └── MLProject/
│       ├── modelling.py
│       ├── conda.yaml
│       └── MLProject
└── monitoring/
    ├── prometheus.yml
    ├── prometheus_exporter.py
    └── inference.py
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

### 3. Run Automated Preprocessing
```bash
cd experimentation/preprocessing
python automate.py
```

### 4. Train Model
```bash
cd modelling

# Baseline (autolog)
python modelling.py

# With hyperparameter tuning (manual logging)
python modelling_tuning.py
```

### 5. Open MLflow UI
```bash
mlflow ui
# Open http://127.0.0.1:5000
```

### 6. Serve Model
```bash
mlflow models serve \
  -m mlruns/<experiment_id>/<run_id>/artifacts/model \
  -p 5001 \
  --env-manager=local \
  --no-conda
```

### 7. Run Monitoring Stack
```bash
# Terminal 1 - Prometheus Exporter
cd monitoring
python prometheus_exporter.py

# Terminal 2 - Prometheus (Docker)
docker run -d \
  --name prometheus \
  -p 9090:9090 \
  -v $(pwd)/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Terminal 3 - Grafana (Docker) with SMTP for alerting
# WARNING: Never put your real App Password in a public repo!
# Use environment variables or secrets management instead.
docker run -d \
  --name grafana \
  -p 3000:3000 \
  -v grafana-storage:/var/lib/grafana \
  -e GF_SMTP_ENABLED=true \
  -e GF_SMTP_HOST=smtp.gmail.com:587 \
  -e GF_SMTP_USER=yourmail@gmail.com \
  -e GF_SMTP_PASSWORD=xxxxxxxxxxxxxx \
  -e GF_SMTP_FROM_ADDRESS=yourmail@gmail.com \
  grafana/grafana

# Open http://localhost:3000 (default: admin/admin)
```

> **Note on Grafana persistence**: The `-v grafana-storage:/var/lib/grafana` flag ensures your dashboards, alert rules, and contact points are preserved even if the container is recreated.

> **Gmail App Password**: To use Gmail SMTP, enable 2-Step Verification on your Google account, then generate an App Password at [myaccount.google.com](https://myaccount.google.com) → Security → App passwords.

### 8. Run Inference
```bash
cd monitoring
python inference.py
```

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

---

## Author

**Ni Putu Sintia Wati**
- GitHub: [@sintiasnn](https://github.com/sintiasnn)

---

## License

This project is open source and available under the [MIT License](LICENSE).