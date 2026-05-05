FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn==1.8.0 pandas==2.3.3 numpy==2.4.4

COPY model/ /app/model/

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "/app/model", "-p", "5001", "--host", "0.0.0.0", "--no-conda"]
