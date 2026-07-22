import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)


# ==========================
# MLflow
# ==========================
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment(
    "Hotel Reservation Classification"
)

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("hotel_reservation_processed.csv")

print(f"Dataset berhasil dibaca: {df.shape}")


# ==========================
# Feature & Target
# ==========================

X = df.drop(columns=["booking_status"])
y = df["booking_status"]


# Encode fitur kategori
X = pd.get_dummies(X)


# Encode target
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)


print("Encoding selesai")


# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training model...")


# ==========================
# MLflow Run
# ==========================

with mlflow.start_run():

    n_estimators = 100
    random_state = 42


    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )


    # Training
    model.fit(
        X_train,
        y_train
    )


    # Prediction
    y_pred = model.predict(X_test)



    # ==========================
    # Evaluation
    # ==========================

    acc = accuracy_score(
        y_test,
        y_pred
    )

    prec = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    rec = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # ==========================
    # MLflow Parameter
    # ==========================

    mlflow.log_param(
        "model",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "random_state",
        random_state
    )


    # ==========================
    # MLflow Metrics
    # ==========================

    mlflow.log_metric(
        "accuracy",
        acc
    )

    mlflow.log_metric(
        "precision",
        prec
    )

    mlflow.log_metric(
        "recall",
        rec
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )



    # ==========================
    # Classification Report
    # ==========================

    report = classification_report(
        y_test,
        y_pred
    )


    with open(
        "classification_report.txt",
        "w"
    ) as f:
        f.write(report)


    mlflow.log_artifact(
        "classification_report.txt"
    )



    # ==========================
    # Confusion Matrix
    # ==========================

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )


    disp.plot()

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix.png"
    )

    plt.close()


    mlflow.log_artifact(
        "confusion_matrix.png"
    )



    # ==========================
    # Feature Importance
    # ==========================

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(
        ascending=False
    )


    plt.figure(
        figsize=(10,6)
    )

    importance.head(15).plot(
        kind="bar"
    )


    plt.title(
        "Top 15 Feature Importance"
    )


    plt.tight_layout()


    plt.savefig(
        "feature_importance.png"
    )

    plt.close()


    mlflow.log_artifact(
        "feature_importance.png"
    )



    # ==========================
    # Save Model
    # ==========================

    joblib.dump(
        model,
        "model.pkl"
    )


    joblib.dump(
        X.columns.tolist(),
        "features.pkl"
    )


    joblib.dump(
        target_encoder,
        "target_encoder.pkl"
    )



    # ==========================
    # Log Model MLflow
    # ==========================

    mlflow.sklearn.log_model(
        model,
        "model"
    )


    print("\n===== HASIL EVALUASI =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")



print("\nTraining selesai.")
print("Model berhasil disimpan:")
print("- model.pkl")
print("- features.pkl")
print("- target_encoder.pkl")
print("\nMLflow : http://127.0.0.1:5000")