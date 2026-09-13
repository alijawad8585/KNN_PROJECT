# KNN Loan Approval Prediction Project

This repository contains:

- A **KNN training pipeline** for the Kaggle Loan Prediction dataset
- A **FastAPI backend** that serves the trained model
- A **Next.js frontend** for real-time loan approval predictions

## 1) Dataset setup

Download `train.csv` from Kaggle:
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Place it at:

```bash
/home/runner/work/KNN_PROJECT/KNN_PROJECT/data/train.csv
```

## 2) Train and evaluate KNN

```bash
cd /home/runner/work/KNN_PROJECT/KNN_PROJECT/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_knn.py --data-path ../data/train.csv --output-dir ./model
```

What the script does:

- Loads data and prints first rows
- Checks missing values
- Fills missing numeric values with median and categorical values with mode
- Encodes categorical columns
- Splits into 80/20 train/test
- Standardizes numerical features
- Trains default KNN
- Experiments with `k = 3, 5, 7`
- Tunes `n_neighbors` with GridSearchCV
- Prints accuracy and confusion matrix for default and tuned models
- Saves:
  - `backend/model/knn_pipeline.joblib`
  - `backend/model/metrics.json`
  - `backend/model/knn_confusion_matrices.png`

## 3) Run FastAPI backend

```bash
cd /home/runner/work/KNN_PROJECT/KNN_PROJECT/backend
source .venv/bin/activate
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API endpoints:

- `GET /health`
- `POST /predict`

## 4) Run Next.js frontend

```bash
cd /home/runner/work/KNN_PROJECT/KNN_PROJECT/frontend
npm install
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 npm run dev
```

Open:

```text
http://localhost:3000
```

Use the form to submit applicant details and see `Approved` or `Rejected` predictions.
