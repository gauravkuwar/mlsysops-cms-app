import os
import time
import threading
import torch
import mlflow.pyfunc
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer
from mlflow.tracking import MlflowClient

app = Flask(__name__)

# Config
mock_config = {
    "max_len": 128,
    "model_name": "google/bert_uncased_L-2_H-128_A-2"
}

MODEL_NAME = "mlsysops-cms-model"
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Staging")

# Initialize components
tokenizer = AutoTokenizer.from_pretrained(mock_config["model_name"])
client = MlflowClient()

# Load model
def load_model_from_stage():
    version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE.lower())
    model_uri = f"models:/{MODEL_NAME}/{version_info.version}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    return model, version_info.version

model, current_version = load_model_from_stage()
latest_version = current_version

# Polling thread
def poll_model_version():
    global model, current_version, latest_version
    while True:
        try:
            version_info = client.get_model_version_by_alias(MODEL_NAME, MODEL_STAGE.lower())
            if version_info.version != latest_version:
                latest_version = version_info.version
                print(f"[Model Update Detected] New version: {latest_version}, reloading...")
                model, current_version = load_model_from_stage()
                print(f"[Model Reloaded] Now using version: {current_version}")
        except Exception as e:
            print(f"[Polling Error] {e}")
        time.sleep(30)

threading.Thread(target=poll_model_version, daemon=True).start()

# Prediction
def model_predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=mock_config["max_len"])
    with torch.no_grad():
        logits = model(**inputs).logits.squeeze().item()
        prob = torch.sigmoid(torch.tensor(logits)).item()
    label = "Non-Toxic" if prob < 0.5 else "Toxic"
    return label, prob

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    label, score = model_predict(text)
    return jsonify({"label": label, "confidence": round(score, 2)})

# (Optional for next step)
@app.route('/healthz', methods=['GET'])
def health():
    return jsonify({"model_version": current_version})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
