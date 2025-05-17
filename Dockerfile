# Use an official slim Python image
FROM python:3.11-slim-buster

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /app

# Install Python packages (with PyTorch CPU wheels)
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Copy the full app
COPY . /app

# Expose Flask port
EXPOSE 8000

# Start the Flask app
CMD ["python", "app.py"]
