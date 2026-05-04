import time
import requests
import json
from prometheus_client import start_http_server, Counter, Histogram, Gauge
import random

# ========================
# Metrics Definition
# ========================

# Counter: total number of prediction requests
PREDICTION_REQUESTS_TOTAL = Counter(
    'prediction_requests_total',
    'Total number of prediction requests to the model'
)

# Counter: number of predictions per class
PREDICTION_CLASS_TOTAL = Counter(
    'prediction_class_total',
    'Total predictions per class',
    ['class_label']
)

# Histogram: prediction request latency
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Prediction request latency in seconds',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# Gauge: model status (1 = up, 0 = down)
MODEL_STATUS = Gauge(
    'model_status',
    'Model serving status (1=up, 0=down)'
)

# Gauge: total positive predictions (sick)
PREDICTION_POSITIVE = Gauge(
    'prediction_positive_total',
    'Total positive predictions (heart disease)'
)

# Gauge: total negative predictions (not sick)
PREDICTION_NEGATIVE = Gauge(
    'prediction_negative_total',
    'Total negative predictions (no heart disease)'
)

# Gauge: simulated model accuracy
MODEL_ACCURACY = Gauge(
    'model_accuracy',
    'Model accuracy (simulated)'
)

# ========================
# Configuration
# ========================
MODEL_URL = "http://127.0.0.1:5001/invocations"
HEADERS = {"Content-Type": "application/json"}

# Sample data for request simulation
SAMPLE_DATA = [
    {"data": [[0.9, -0.5, -0.2, 0.3, -0.7, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]]},
    {"data": [[-0.5, 0.3, 0.8, -0.4, 0.2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]]},
    {"data": [[0.2, -0.1, -0.5, 0.7, -0.3, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]]},
    {"data": [[-0.8, 0.6, 0.3, -0.2, 0.9, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1]]},
    {"data": [[1.2, -0.8, -0.1, 0.5, -0.6, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0]]},
]

COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex_1", "cp_2", "cp_3", "cp_4", "fbs_1",
    "restecg_1", "restecg_2", "exang_1", "slope_2", "slope_3",
    "ca_1.0", "ca_2.0", "ca_3.0", "thal_6.0", "thal_7.0"
]

positive_count = 0
negative_count = 0


def predict(data):
    """Send prediction request to model and record metrics."""
    global positive_count, negative_count

    payload = {
        "dataframe_split": {
            "columns": COLUMNS,
            "data": data["data"]
        }
    }

    start_time = time.time()
    try:
        response = requests.post(MODEL_URL, headers=HEADERS, json=payload, timeout=5)
        latency = time.time() - start_time

        # Record metrics
        PREDICTION_REQUESTS_TOTAL.inc()
        PREDICTION_LATENCY.observe(latency)
        MODEL_STATUS.set(1)

        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions'][0]

            if prediction == 1:
                positive_count += 1
                PREDICTION_CLASS_TOTAL.labels(class_label='heart_disease').inc()
            else:
                negative_count += 1
                PREDICTION_CLASS_TOTAL.labels(class_label='no_heart_disease').inc()

            PREDICTION_POSITIVE.set(positive_count)
            PREDICTION_NEGATIVE.set(negative_count)

            # Simulate accuracy
            MODEL_ACCURACY.set(round(random.uniform(0.72, 0.82), 4))

            print(f"    Prediction: {prediction} | Latency: {latency:.3f}s | +:{positive_count} -:{negative_count}")
        else:
            MODEL_STATUS.set(0)
            print(f"    Error: {response.status_code}")

    except Exception as e:
        latency = time.time() - start_time
        MODEL_STATUS.set(0)
        print(f"    Exception: {e}")


def main():
    # Run Prometheus metrics server on port 8000
    start_http_server(8000)
    print("=" * 50)
    print("  PROMETHEUS EXPORTER RUNNING")
    print("  Metrics available at: http://localhost:8000/metrics")
    print("=" * 50)

    while True:
        # Choose random sample data and send prediction
        data = random.choice(SAMPLE_DATA)
        predict(data)
        time.sleep(5)  # Send request every 5 seconds


if __name__ == "__main__":
    main()