FROM python:3.11-slim-buster

WORKDIR /app

# Install system dependencies (no recommended extras)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    onnxruntime \
    numpy \
    mlflow \
    transformers \
    scikit-learn \
    Flask

# Copy application code
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
