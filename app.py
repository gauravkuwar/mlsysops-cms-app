import torch
from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig

app = Flask(__name__)

# Config
mock_config = {
    "max_len": 128,
    "model_name": "google/bert_uncased_L-2_H-128_A-2"
}

tokenizer = AutoTokenizer.from_pretrained(mock_config["model_name"])
config = BertConfig.from_pretrained(mock_config["model_name"], num_labels=1)
model = AutoModelForSequenceClassification.from_config(config)
state_dict = torch.load("model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)