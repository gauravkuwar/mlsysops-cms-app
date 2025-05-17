FROM python:3.11-slim-buster

WORKDIR /app

# Install system dependencies (no recommended extras)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    mlflow \
    transformers \
    scikit-learn \
    Flask

# Copy the rest of the app
COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
