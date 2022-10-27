from flask import Flask, jsonify, request
from main import predictor

app = Flask.app()

@app.route("/predict", methods=["POST"])
def predict():
    img = request.files.get("alphabet")
    prediction = predictor(img)
    return jsonify({
        "prediction": prediction
    }), 200