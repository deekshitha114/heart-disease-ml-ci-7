
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY heart.csv .
COPY train_model.py .
COPY quality_gate.py .
COPY app.py .
COPY test_app.py .
COPY heart_model.pkl .
COPY features.json .
COPY model_metrics.json .

EXPOSE 5000

CMD ["python", "app.py"]
