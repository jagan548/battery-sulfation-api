from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__Rejuvespark__)

model = joblib.load("sulfation_model.joblib")   # FIXED
scaler = joblib.load("input_scaler.joblib")     # FIXED

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    voltage = data['voltage']
    current = data['current']
    percentage = data['percentage']

    scaled = scaler.transform([[voltage, current, percentage]])
    pred = model.predict(scaled)[0]
    
    return jsonify({"sulfation_level": float(pred)})

app.run(host='0.0.0.0', port=8000)
