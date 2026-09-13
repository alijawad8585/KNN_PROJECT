# KNN Loan Approval Prediction Project

This repository implements the assignment requirements for predicting loan approval using a K-Nearest Neighbors (KNN) model, plus a Flask API backend and Next.js frontend.

## 1) Dataset

- Primary input file: `data/train.csv` (download from Kaggle: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- If `data/train.csv` is not found, the training script falls back to a public mirror URL.

## 2) Backend + Model Training

### Setup

```bash
cd /home/runner/work/KNN_PROJECT/KNN_PROJECT/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Train and evaluate KNN

```bash
python train_knn.py
```

The script:
- loads data and handles missing values (median for numeric, mode for categorical)
- encodes categorical features (OneHotEncoder)
- splits data 80/20
- standardizes numeric features (StandardScaler)
- trains default KNN
- evaluates k in `{3, 5, 7}`
- tunes `n_neighbors` with `GridSearchCV`
- prints accuracy + confusion matrix for default and tuned models
- saves model artifact to `backend/artifacts/knn_pipeline.joblib`
- saves confusion matrix plot to `backend/artifacts/confusion_matrices.png`

### Run Flask API

```bash
python app.py
```

API endpoints:
- `GET /health`
- `POST /predict` with JSON body:

```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "0",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 0,
  "LoanAmount": 120,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

## 3) Next.js Frontend

```bash
cd /home/runner/work/KNN_PROJECT/KNN_PROJECT/frontend
npm install
npm run dev
```

By default, frontend requests backend at `http://127.0.0.1:8000`.
Set `NEXT_PUBLIC_API_URL` to override.
