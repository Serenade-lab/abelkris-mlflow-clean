from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# =========================
# LOAD MODEL & PREPROCESSING
# =========================
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")
encoder = joblib.load("target_encoder.pkl")


# =========================
# HOME CHECK
# =========================
@app.route("/")
def home():
    return "Hotel Reservation Prediction API is Running"


# =========================
# PREDICTION API
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil JSON dari request
        data = request.get_json()

        # Ubah ke dataframe
        df = pd.DataFrame([data])

        # Encoding kategori
        df_encoded = encoder.transform(df)

        # Jadikan dataframe
        df_encoded = pd.DataFrame(
            df_encoded,
            columns=features
        )

        # Prediksi model
        prediction = model.predict(df_encoded)

        # Label hasil
        label = (
            "Canceled"
            if prediction[0] == 1
            else "Not_Canceled"
        )

        return jsonify({
            "prediction": int(prediction[0]),
            "label": label
        })


    except Exception as e:
        return jsonify({
            "error": str(e)
        })


# =========================
# RUN FLASK
# =========================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000
    )