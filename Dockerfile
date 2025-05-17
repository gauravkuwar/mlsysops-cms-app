# Use an official slim Python image
FROM python:3.11-slim-buster

# Set working directory early
WORKDIR /app

# Install only needed system packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt .

# Use --prefer-binary to avoid building wheels
RUN pip install --no-cache-dir --prefer-binary \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Now copy the rest of the app
COPY . .

# Expose Flask port
EXPOSE 8000

# Run app
CMD ["python", "app.py"]