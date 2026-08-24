"""
app.py — Flask web app that serves the car price prediction model.

Run locally:
    python app.py
Then open http://127.0.0.1:5000
"""

import os
import json
import pickle
from datetime import datetime

import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METADATA_PATH = os.path.join(BASE_DIR, "model_metadata.json")

# ---------------------------------------------------------------
# Load model + metadata once at startup
# ---------------------------------------------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

CATEGORICAL_VALUES = metadata["categorical_values"]
CURRENT_YEAR = datetime.now().year


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        fuel_types=CATEGORICAL_VALUES["Fuel_Type"],
        selling_types=CATEGORICAL_VALUES["Selling_type"],
        transmissions=CATEGORICAL_VALUES["Transmission"],
        current_year=CURRENT_YEAR,
        prediction=None,
    )


def build_features_from_form(form):
    year = int(form["year"])
    present_price = float(form["present_price"])
    driven_kms = int(form["driven_kms"])
    fuel_type = form["fuel_type"]
    selling_type = form["selling_type"]
    transmission = form["transmission"]
    owner = int(form["owner"])

    car_age = CURRENT_YEAR - year

    row = pd.DataFrame([{
        "Present_Price": present_price,
        "Driven_kms": driven_kms,
        "Fuel_Type": fuel_type,
        "Selling_type": selling_type,
        "Transmission": transmission,
        "Owner": owner,
        "Car_Age": car_age,
    }])
    return row


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = build_features_from_form(request.form)
        pred = model.predict(row)[0]
        pred = max(0, round(float(pred), 2))  # price can't be negative

        return render_template(
            "index.html",
            fuel_types=CATEGORICAL_VALUES["Fuel_Type"],
            selling_types=CATEGORICAL_VALUES["Selling_type"],
            transmissions=CATEGORICAL_VALUES["Transmission"],
            current_year=CURRENT_YEAR,
            prediction=pred,
        )
    except Exception as e:
        return render_template(
            "index.html",
            fuel_types=CATEGORICAL_VALUES["Fuel_Type"],
            selling_types=CATEGORICAL_VALUES["Selling_type"],
            transmissions=CATEGORICAL_VALUES["Transmission"],
            current_year=CURRENT_YEAR,
            prediction=None,
            error=str(e),
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint, e.g. for testing with curl/Postman."""
    try:
        data = request.get_json(force=True)
        row = pd.DataFrame([{
            "Present_Price": float(data["present_price"]),
            "Driven_kms": int(data["driven_kms"]),
            "Fuel_Type": data["fuel_type"],
            "Selling_type": data["selling_type"],
            "Transmission": data["transmission"],
            "Owner": int(data["owner"]),
            "Car_Age": CURRENT_YEAR - int(data["year"]),
        }])
        pred = model.predict(row)[0]
        pred = max(0, round(float(pred), 2))
        return jsonify({"predicted_selling_price_lakhs": pred})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # debug=True is fine locally; turn it off in production (handled by gunicorn)
    app.run(debug=True)
