from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Hotel Reservation Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data JSON
        data = request.get_json()

        # Ubah menjadi DataFrame
        df = pd.DataFrame([data])

        # Prediksi
        prediction = model.predict(df)

        # Konversi hasil prediksi ke label
        label = "Canceled" if prediction[0] == 1 else "Not_Canceled"

        return jsonify({
            "prediction": int(prediction[0]),
            "label": label
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)