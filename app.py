from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("model.joblib")


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Performance ML API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "Send a JSON object"}), 400

    try:
        study_hours = float(data["study_hours"])
        attendance = float(data["attendance"])
        previous_score = float(data["previous_score"])
    except (KeyError, TypeError, ValueError):
        return jsonify({
            "error": "Provide study_hours, attendance, and previous_score as numbers"
        }), 400

    if (
        not 0 <= study_hours <= 24
        or not 0 <= attendance <= 100
        or not 0 <= previous_score <= 100
    ):
        return jsonify({
            "error": "Study hours must be 0–24; attendance and score must be 0–100"
        }), 400

    sample = pd.DataFrame(
        [[study_hours, attendance, previous_score]],
        columns=["study_hours", "attendance", "previous_score"]
    )

    result = int(model.predict(sample)[0])

    return jsonify({
        "prediction": "Pass" if result == 1 else "Needs support",
        "demo_model": True,
        "notice": "Experimental prediction using illustrative training data"
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)