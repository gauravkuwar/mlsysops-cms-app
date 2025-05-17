# Use an official slim Python image
FROM python:3.11-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app

# Install Python packages (PyTorch + MLflow + Transformers)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy application code
COPY . /app

# Expose Flask port
EXPOSE 8000

# Run the Flask app
CMD ["python", "app.py"]
