import joblib
from pathlib import Path


MODEL_DIR = Path("../models")

MODEL_DIR.mkdir(
    exist_ok=True
)


joblib.dump(
    kmeans,
    MODEL_DIR / "kmeans_model.joblib"
)


joblib.dump(
    scaler,
    MODEL_DIR / "scaler.joblib"
)