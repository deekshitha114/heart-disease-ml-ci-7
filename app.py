
from flask import Flask, request, jsonify
import pickle
import json
import pandas as pd

app = Flask(__name__)

with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("features.json") as f:
    features = json.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/predict", methods=["POST"])
def predict():
    values = request.get_json(silent=True)

    if not isinstance(values, dict):
        return jsonify({"error": "JSON object required"}), 400

    if not all(feature in values for feature in features):
        return jsonify({"error": "Missing input features"}), 400

    try:
        input_data = pd.DataFrame(
            [[values[name] for name in features]],
            columns=features
        )
        prediction = int(model.predict(input_data)[0])

        return jsonify({"prediction": prediction})

    except (TypeError, ValueError):
        return jsonify({"error": "Invalid input values"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
