from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "model" / "knn_pipeline.joblib"
LABEL_MAP = {0: "Rejected", 1: "Approved"}


class LoanApplication(BaseModel):
    Gender: str = Field(..., examples=["Male", "Female"])
    Married: str = Field(..., examples=["Yes", "No"])
    Dependents: str = Field(..., examples=["0", "1", "2", "3+"])
    Education: str = Field(..., examples=["Graduate", "Not Graduate"])
    Self_Employed: str = Field(..., examples=["Yes", "No"])
    ApplicantIncome: float = Field(..., ge=0)
    CoapplicantIncome: float = Field(..., ge=0)
    LoanAmount: float = Field(..., ge=0)
    Loan_Amount_Term: float = Field(..., gt=0)
    Credit_History: float = Field(..., ge=0, le=1)
    Property_Area: str = Field(..., examples=["Urban", "Semiurban", "Rural"])


app = FastAPI(title="Loan Approval Predictor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. Run backend/train_knn.py first."
        )
    app.state.model = joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(application: LoanApplication) -> dict:
    model = getattr(app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    payload = [application.model_dump()]
    prediction = int(model.predict(payload)[0])
    return {
        "prediction": prediction,
        "result": LABEL_MAP[prediction],
    }
