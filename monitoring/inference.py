import requests
import json

# ========================
# Configuration
# ========================
MODEL_URL = "http://127.0.0.1:5001/invocations"
HEADERS = {"Content-Type": "application/json"}

COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak",
    "sex_1", "cp_2", "cp_3", "cp_4", "fbs_1",
    "restecg_1", "restecg_2", "exang_1", "slope_2", "slope_3",
    "ca_1.0", "ca_2.0", "ca_3.0", "thal_6.0", "thal_7.0"
]


def predict(data: list) -> dict:
    """
    Predict using the served ML model.

    Args:
        data (list): List of feature values following the COLUMNS order

    Returns:
        dict: Model response containing predictions
    """
    payload = {
        "dataframe_split": {
            "columns": COLUMNS,
            "data": [data]
        }
    }

    try:
        response = requests.post(MODEL_URL, headers=HEADERS, json=payload, timeout=5)
        if response.status_code == 200:
            result = response.json()
            prediction = result['predictions'][0]
            label = "Heart Disease" if prediction == 1 else "No Heart Disease"
            return {
                "status": "success",
                "prediction": prediction,
                "label": label
            }
        else:
            return {
                "status": "error",
                "message": f"HTTP {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


if __name__ == "__main__":
    print("=" * 50)
    print("  INFERENCE - HEART DISEASE PREDICTION")
    print("=" * 50)

    # Sample patient data
    sample_patients = [
        {
            "name": "Patient A (likely sick)",
            "data": [0.9, -0.5, -0.2, 0.3, -0.7, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
        },
        {
            "name": "Patient B (likely healthy)",
            "data": [-0.5, 0.3, 0.8, -0.4, 0.2, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
        },
        {
            "name": "Patient C",
            "data": [0.2, -0.1, -0.5, 0.7, -0.3, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]
        }
    ]

    for patient in sample_patients:
        print(f"\n>>> {patient['name']}")
        result = predict(patient['data'])
        if result['status'] == 'success':
            print(f"    Prediction : {result['prediction']}")
            print(f"    Label      : {result['label']}")
        else:
            print(f"    Error      : {result['message']}")

    print("\n" + "=" * 50)
    print("  INFERENCE DONE")
    print("=" * 50)