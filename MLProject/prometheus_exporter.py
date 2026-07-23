from prometheus_client import start_http_server, Counter, Gauge
import pandas as pd
import joblib
import time


# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")


# =========================
# METRICS
# =========================
prediction_total = Counter(
    "prediction_total",
    "Total prediction request"
)

prediction_result = Gauge(
    "prediction_result",
    "Latest prediction result"
)


# =========================
# SAMPLE INPUT
# =========================
sample = pd.DataFrame([{
    "lead_time": 100,
    "no_of_special_requests": 1,
    "avg_price_per_room": 100,
    "no_of_adults": 2,
    "no_of_weekend_nights": 1,
    "no_of_week_nights": 2,
    "type_of_meal_plan_Meal Plan 1": 1,
    "type_of_meal_plan_Meal Plan 2": 0,
    "type_of_meal_plan_Not Selected": 0,
    "room_type_reserved_Room_Type 1": 1,
    "room_type_reserved_Room_Type 2": 0,
    "room_type_reserved_Room_Type 3": 0,
    "market_segment_type_Online": 1,
    "market_segment_type_Offline": 0,
    "market_segment_type_Corporate": 0
}])


if __name__ == "__main__":

    start_http_server(8000)

    print(
        "Prometheus Exporter running at http://localhost:8000/metrics"
    )


    while True:

        try:

            # samakan kolom dengan training
            sample = sample.reindex(
                columns=features,
                fill_value=0
            )


            prediction = model.predict(sample)


            prediction_total.inc()

            prediction_result.set(
                int(prediction[0])
            )


            print(
                "Prediction:",
                prediction[0]
            )


        except Exception as e:
            print(
                "Error:",
                e
            )


        time.sleep(5)