from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # ✅ VERY IMPORTANT

model = joblib.load("sulfation_model.joblib")
scaler = joblib.load("input_scaler.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    voltage = float(data['voltage'])
    current = float(data['current'])
    percentage = float(data['percentage'])

    # Scale inputs
    scaled = scaler.transform([[voltage, current, percentage]])

    # Model output (Efficiency)
    efficiency = float(model.predict(scaled)[0])

    # Sulfation = 100 - efficiency
    sulfation = 100 - efficiency

    return jsonify({
        "efficiency": round(efficiency, 2),
        "sulfation": round(sulfation, 2),
    })
