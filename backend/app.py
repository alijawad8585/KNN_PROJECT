from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from knn_pipeline import ARTIFACT_PATH, load_artifact, save_artifact, train_and_evaluate

app = Flask(__name__)
CORS(app)


class ModelService:
    def __init__(self, artifact_path: Path = ARTIFACT_PATH) -> None:
        self.artifact_path = artifact_path
        self.bundle = self._load_or_train()

    def _load_or_train(self):
        if self.artifact_path.exists():
            return load_artifact(self.artifact_path)

        training_results = train_and_evaluate()
        metadata = {
            "best_n_neighbors": training_results["best_n_neighbors"],
            "default_metrics": training_results["default_metrics"],
            "tuned_metrics": training_results["tuned_metrics"],
            "k_experiments": training_results["k_experiments"],
        }
        save_artifact(training_results["tuned_model"], metadata, self.artifact_path)
        return {"model": training_results["tuned_model"], "metadata": metadata}

    @property
    def model(self):
        return self.bundle["model"]


service = ModelService()
REQUIRED_FIELDS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area",
]
NUMERIC_FIELDS = {
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
}


@app.get("/health")
def health_check():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict_loan_status():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON payload."}), 400

    missing_fields = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    formatted_payload = {}
    for field in REQUIRED_FIELDS:
        value = payload[field]
        if field in NUMERIC_FIELDS:
            try:
                formatted_payload[field] = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": f"Field '{field}' must be numeric."}), 400
        else:
            formatted_payload[field] = str(value)

    input_frame = pd.DataFrame([formatted_payload])
    prediction = int(service.model.predict(input_frame)[0])
    label = "Y" if prediction == 1 else "N"
    decision = "Approved" if prediction == 1 else "Rejected"

    return jsonify({"prediction": decision, "label": label})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
