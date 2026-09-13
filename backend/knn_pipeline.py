from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv"
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "train.csv"
ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "knn_pipeline.joblib"
PLOT_PATH = Path(__file__).resolve().parent / "artifacts" / "confusion_matrices.png"

FEATURE_COLUMNS = [
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
TARGET_COLUMN = "Loan_Status"
NUMERIC_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
CATEGORICAL_COLUMNS = [c for c in FEATURE_COLUMNS if c not in NUMERIC_COLUMNS]


def load_dataset(data_path: Path | None = None) -> pd.DataFrame:
    path = data_path or DEFAULT_DATA_PATH
    if path.exists():
        return pd.read_csv(path)
    return pd.read_csv(DATA_URL)


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )


def _evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }


def train_and_evaluate(data_path: Path | None = None) -> dict[str, Any]:
    df = load_dataset(data_path)

    x = df[FEATURE_COLUMNS].copy()
    y_raw = df[TARGET_COLUMN]
    if y_raw.dtype.kind in {"i", "u", "f"}:
        y = y_raw.astype(int)
    else:
        y = y_raw.astype(str).str.strip().str.upper().map({"N": 0, "Y": 1})

    valid_target = y.notna()
    x = x.loc[valid_target]
    y = y.loc[valid_target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = _build_preprocessor()

    default_knn = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier()),
        ]
    )
    default_knn.fit(x_train, y_train)
    default_metrics = _evaluate_model(default_knn, x_test, y_test)

    k_experiments: dict[int, float] = {}
    for k in (3, 5, 7):
        experiment_model = Pipeline(
            steps=[
                ("preprocessor", _build_preprocessor()),
                ("classifier", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        experiment_model.fit(x_train, y_train)
        k_experiments[k] = _evaluate_model(experiment_model, x_test, y_test)["accuracy"]

    tune_base_model = Pipeline(
        steps=[
            ("preprocessor", _build_preprocessor()),
            ("classifier", KNeighborsClassifier()),
        ]
    )
    grid_search = GridSearchCV(
        tune_base_model,
        param_grid={"classifier__n_neighbors": list(range(1, 21))},
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)
    tuned_knn = grid_search.best_estimator_
    tuned_metrics = _evaluate_model(tuned_knn, x_test, y_test)

    return {
        "default_model": default_knn,
        "tuned_model": tuned_knn,
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "k_experiments": k_experiments,
        "best_n_neighbors": int(grid_search.best_params_["classifier__n_neighbors"]),
    }


def plot_confusion_matrices(
    default_cm: list[list[int]], tuned_cm: list[list[int]], output_path: Path = PLOT_PATH
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, matrix, title in (
        (axes[0], default_cm, "Default KNN"),
        (axes[1], tuned_cm, "Tuned KNN"),
    ):
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], labels=["Rejected", "Approved"])
        ax.set_yticks([0, 1], labels=["Rejected", "Approved"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_artifact(model: Pipeline, metadata: dict[str, Any], artifact_path: Path = ARTIFACT_PATH) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, artifact_path)


def load_artifact(artifact_path: Path = ARTIFACT_PATH) -> dict[str, Any]:
    return joblib.load(artifact_path)
