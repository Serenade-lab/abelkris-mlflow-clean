from prometheus_client import start_http_server, Counter, Gauge
import time
import joblib
import pandas as pd

# ==========================
# Load Model
# ==========================
model = joblib.load("model.pkl")

# ==========================
# Metrics
# ==========================
REQUEST_COUNT = Counter(
    "model_requests_total",
    "Total number of prediction requests"
)

MODEL_ACCURACY = Gauge(
    "model_accuracy",
    "Model accuracy"
)

PREDICTION_TIME = Gauge(
    "prediction_time_seconds",
    "Prediction latency"
)

PREDICTION_RESULT = Gauge(
    "prediction_result",
    "Prediction result (0=Not_Canceled, 1=Canceled)"
)

# ==========================
# Sample Data
# (sesuaikan dengan hasil preprocessing)
# ==========================
sample = pd.DataFrame({
    "no_of_adults": [2],
    "no_of_children": [0],
    "no_of_weekend_nights": [1],
    "no_of_week_nights": [2],
    "type_of_meal_plan": [0],
    "required_car_parking_space": [0],
    "room_type_reserved": [1],
    "lead_time": [45],
    "arrival_year": [2018],
    "arrival_month": [7],
    "arrival_date": [15],
    "market_segment_type": [1],
    "repeated_guest": [0],
    "no_of_previous_cancellations": [0],
    "no_of_previous_bookings_not_canceled": [0],
    "avg_price_per_room": [110.5],
    "no_of_special_requests": [1]
})

# ==========================
# Main
# ==========================
if __name__ == "__main__":

    print("Prometheus Exporter running at http://localhost:8000/metrics")

    start_http_server(8000)

    while True:

        start = time.time()

        prediction = model.predict(sample)

        latency = time.time() - start

        REQUEST_COUNT.inc()

        # Ganti sesuai hasil accuracy model
        MODEL_ACCURACY.set(0.90)

        PREDICTION_TIME.set(latency)

        PREDICTION_RESULT.set(int(prediction[0]))

        print(
            f"Prediction: {prediction[0]} | "
            f"Latency: {latency:.5f} sec"
        )

        time.sleep(5)