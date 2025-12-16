from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("sulfation_model.joblib")
scaler = joblib.load("input_scaler.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    voltage = data['voltage']
    current = data['current']
    percentage = data['percentage']

    scaled = scaler.transform([[voltage, current, percentage]])
    pred = model.predict(scaled)[0]

    efficiency = float(pred)
    sulfation = 100 - efficiency

    return jsonify({
        "efficiency": round(efficiency, 2),
        "sulfation": round(sulfation, 2)
    })

app.run(host='0.0.0.0', port=8000)
