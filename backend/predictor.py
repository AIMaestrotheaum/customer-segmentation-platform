from pathlib import Path
import joblib

from backend.preprocessing import prepare_features


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

MODEL_PATH = MODEL_DIR / "kmeans_model.joblib"

SCALER_PATH = MODEL_DIR / "scaler.joblib"


try:

    kmeans_model = joblib.load(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    print("Model and scaler loaded successfully.")

except FileNotFoundError:

    kmeans_model = None
    scaler = None

    print(
        "WARNING: Model files not found."
    )


# ------------------------------------------------------------
# Segment names
# ------------------------------------------------------------

SEGMENTS = {

    0: "Mainstream Customers",

    1: "High-Value Support-Intensive Customers"

}


# ------------------------------------------------------------
# Recommendations
# ------------------------------------------------------------

RECOMMENDATIONS = {

    0:
        "Use targeted engagement, retention, "
        "cross-selling, and personalized recommendations.",

    1:
        "Focus on retention, premium products, "
        "cross-selling, and personalized offers. "
        "Monitor the high support interaction rate."
}


# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------

def predict_customer(customer_data: dict):

    if kmeans_model is None or scaler is None:

        raise RuntimeError(
            "Model files are missing. "
            "Train and save the clustering model first."
        )


    # Prepare features
    features = prepare_features(
        customer_data
    )


    # Scale
    scaled_features = scaler.transform(
        features
    )


    # Predict
    cluster = int(
        kmeans_model.predict(
            scaled_features
        )[0]
    )


    # Segment
    segment = SEGMENTS.get(
        cluster,
        f"Cluster {cluster}"
    )


    # Recommendation
    recommendation = RECOMMENDATIONS.get(
        cluster,
        "Use targeted customer engagement."
    )


    return {

        "cluster": cluster,

        "segment": segment,

        "recommendation": recommendation

    }