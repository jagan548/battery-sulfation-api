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

    scaled = scaler.transform([[voltage, current, percentage]])
    sulfation = model.predict(scaled)[0]

    efficiency = 100 - sulfation

    return jsonify({
         "voltage": voltage,
         "current": current,
         "efficiency": float(efficiency),
         "sulfation": float(sulfation),
         "device_status": "CONNECTED"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
