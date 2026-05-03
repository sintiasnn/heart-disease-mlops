# 🫀 Heart Disease MLOps Pipeline

End-to-end MLOps pipeline untuk prediksi penyakit jantung menggunakan dataset Heart Disease dari UCI Machine Learning Repository. Project ini mencakup seluruh siklus MLOps mulai dari eksperimen, pembangunan model, CI/CD workflow, hingga monitoring dan logging.

---

## 📋 Daftar Isi
- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Struktur Project](#struktur-project)
- [Dataset](#dataset)
- [Cara Menjalankan](#cara-menjalankan)
- [Hasil Eksperimen](#hasil-eksperimen)
- [Monitoring](#monitoring)

---

## 🔍 Overview

Project ini membangun pipeline MLOps lengkap untuk memprediksi apakah seorang pasien menderita penyakit jantung atau tidak (binary classification) berdasarkan data klinis.

**Pipeline yang dibangun:**
1. **Eksperimen** — EDA dan preprocessing data secara otomatis
2. **Modelling** — Pelatihan model dengan MLflow tracking
3. **CI/CD** — Otomatisasi retraining dengan GitHub Actions
4. **Monitoring** — Real-time monitoring dengan Prometheus & Grafana

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| Language | Python 3.12 |
| ML Framework | Scikit-Learn |
| Experiment Tracking | MLflow 2.19.0 |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Containerization | Docker |
| Data Source | UCI ML Repository |

---

## 📁 Struktur Project

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
│   ├── MLProject/
│   │   ├── modelling.py
│   │   ├── conda.yaml
│   │   └── MLProject
│   └── .github/
│       └── workflows/
│           └── ci.yml
└── monitoring/
    ├── prometheus.yml
    ├── prometheus_exporter.py
    └── inference.py
```

---

## 📊 Dataset

- **Sumber**: [UCI ML Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Jumlah data**: 303 baris, 13 fitur + 1 target
- **Task**: Binary Classification (0 = tidak sakit, 1 = sakit jantung)
- **Fitur**: Campuran numerikal dan kategorikal (age, sex, cp, trestbps, chol, dll)

---

## 🚀 Cara Menjalankan

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

### 3. Jalankan Preprocessing Otomatis
```bash
cd experimentation/preprocessing
python automate.py
```

### 4. Training Model
```bash
cd modelling

# Baseline (autolog)
python modelling.py

# Dengan hyperparameter tuning (manual logging)
python modelling_tuning.py
```

### 5. Buka MLflow UI
```bash
mlflow ui
# Buka http://127.0.0.1:5000
```

### 6. Serving Model
```bash
mlflow models serve \
  -m mlruns/<experiment_id>/<run_id>/artifacts/model \
  -p 5001 \
  --env-manager=local \
  --no-conda
```

### 7. Jalankan Monitoring
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

# Terminal 3 - Grafana (Docker)
docker run -d \
  --name grafana \
  -p 3000:3000 \
  grafana/grafana

# Buka http://localhost:3000
```

### 8. Inference
```bash
cd monitoring
python inference.py
```

---

## 📈 Hasil Eksperimen

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Random Forest (Baseline) | 0.7368 | 0.7273 | 0.6400 | 0.6809 | — |
| Random Forest (Tuning) | **0.7719** | **0.8000** | 0.6400 | **0.7111** | **0.8575** |

**Best Parameters (GridSearchCV):**
- `n_estimators`: 100
- `max_depth`: 5
- `min_samples_split`: 2

---

## 📡 Monitoring

Sistem monitoring real-time menggunakan Prometheus dan Grafana dengan metrics:

| Metric | Deskripsi |
|---|---|
| `prediction_requests_total` | Total request prediksi |
| `prediction_class_total` | Prediksi per kelas (sakit/tidak sakit) |
| `prediction_latency_seconds` | Latensi request prediksi |
| `model_status` | Status model (1=up, 0=down) |
| `prediction_positive_total` | Total prediksi positif |
| `prediction_negative_total` | Total prediksi negatif |
| `model_accuracy` | Akurasi model |

---

## 👩‍💻 Author

**Ni Putu Sintia Wati**
- GitHub: [@sintiasnn](https://github.com/sintiasnn)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).