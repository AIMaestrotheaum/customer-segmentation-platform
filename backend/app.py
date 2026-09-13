from fastapi import FastAPI, HTTPException

from backend.schemas import (
    CustomerInput,
    PredictionResponse
)

from backend.predictor import (
    predict_customer
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(

    title="Customer Segmentation API",

    description=(
        "Machine learning API for customer "
        "segmentation and behavioral analytics."
    ),

    version="1.0.0"
)


# ============================================================
# Health check
# ============================================================

@app.get("/")
def root():

    return {

        "message":
            "Customer Segmentation API is running",

        "status":
            "healthy"

    }


# ============================================================
# API health endpoint
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok"

    }


# ============================================================
# Prediction endpoint
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    customer: CustomerInput
):

    try:

        result = predict_customer(
            customer.model_dump()
        )

        return result

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )