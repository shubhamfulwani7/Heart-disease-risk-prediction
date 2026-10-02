from flask import Flask, jsonify, request
from flask_cors import CORS

import joblib
import pandas as pd
import json

from pathlib import Path


# ==========================================
# APP SETUP
# ==========================================

app = Flask(
    __name__,
    static_folder="../frontend",
    static_url_path=""
)

CORS(app)


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "heart_model.pkl"

DATA_PATH = BASE_DIR / "data" / "heart.csv"

METRICS_PATH = BASE_DIR / "model" / "metrics.json"


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv(DATA_PATH)

df.columns = df.columns.str.strip()


# ==========================================
# HOME
# ==========================================

@app.route("/")
def home():
    return app.send_static_file("index.html")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.route("/api/health")
def health():

    return jsonify({

        "status": "success",

        "message":
            "Heart Disease Prediction API is running"

    })


# ==========================================
# PREDICTION API
# ==========================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json()


        features = [[

            data["age"],

            data["sex"],

            data["cp"],

            data["trestbps"],

            data["chol"],

            data["fbs"],

            data["restecg"],

            data["thalach"],

            data["exang"],

            data["oldpeak"],

            data["slope"],

            data["ca"],

            data["thal"]

        ]]


        prediction = model.predict(features)[0]


        probability = model.predict_proba(
            features
        )[0][1]


        if prediction == 1:

            result = "Heart Disease Detected"

        else:

            result = "No Heart Disease Detected"


        return jsonify({

            "prediction": int(prediction),

            "result": result,

            "probability": round(
                float(probability) * 100,
                2
            )

        })


    except Exception as error:

        return jsonify({

            "error": str(error)

        }), 400


# ==========================================
# EDA API
# ==========================================

@app.route("/api/eda")
def eda():

    # --------------------------------------
    # TARGET DISTRIBUTION
    # --------------------------------------

    target_counts = (

        df["target"]

        .value_counts()

        .sort_index()

        .to_dict()

    )


    # --------------------------------------
    # AGE
    # --------------------------------------

    age_data = df["age"].tolist()


    # --------------------------------------
    # SEX VS TARGET
    # --------------------------------------

    sex_table = (

        df.groupby(
            ["sex", "target"]
        )

        .size()

        .unstack(fill_value=0)

    )


    sex_target = {}


    for sex in sex_table.index:

        sex_target[str(sex)] = {}


        for target in [0, 1]:

            if target in sex_table.columns:

                value = sex_table.loc[
                    sex,
                    target
                ]

            else:

                value = 0


            sex_target[str(sex)][
                str(target)
            ] = int(value)


    # --------------------------------------
    # CHEST PAIN VS TARGET
    # --------------------------------------

    cp_table = (

        df.groupby(
            ["cp", "target"]
        )

        .size()

        .unstack(fill_value=0)

    )


    cp_target = {}


    for cp in cp_table.index:

        cp_target[str(cp)] = {}


        for target in [0, 1]:

            if target in cp_table.columns:

                value = cp_table.loc[
                    cp,
                    target
                ]

            else:

                value = 0


            cp_target[str(cp)][
                str(target)
            ] = int(value)


    # --------------------------------------
    # CORRELATION
    # --------------------------------------

    correlation = df.corr(
        numeric_only=True
    )


    return jsonify({

        "target": {

            str(key): int(value)

            for key, value
            in target_counts.items()

        },


        "age": age_data,


        "sex_target": sex_target,


        "cp_target": cp_target,


        "correlation": {

            "columns":
                correlation.columns.tolist(),

            "values":
                correlation.values.tolist()

        }

    })


# ==========================================
# MODEL METRICS API
# ==========================================

@app.route("/api/model")
def model_metrics():

    try:

        if not METRICS_PATH.exists():

            return jsonify({

                "error":
                    "metrics.json not found. Run train_model.py first."

            }), 404


        with open(
            METRICS_PATH,
            "r"
        ) as file:

            metrics = json.load(file)


        return jsonify(metrics)


    except Exception as error:

        return jsonify({

            "error": str(error)

        }), 500


# ==========================================
# START SERVER
# ==========================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )