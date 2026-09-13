from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Loan_Status"
DROP_COLUMNS = ["Loan_ID"]
LABEL_MAP = {"N": 0, "Y": 1}


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = features.select_dtypes(exclude=["number"]).columns.tolist()

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
            ("numerical", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def evaluate_model(name: str, model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print(f"\n{name} model")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return {"name": name, "accuracy": acc, "confusion_matrix": cm, "predictions": predictions}


def main(data_path: Path, output_dir: Path) -> None:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}. Place train.csv there and rerun.")

    df = pd.read_csv(data_path)
    print("First 5 rows:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())

    df = df.drop(columns=[col for col in DROP_COLUMNS if col in df.columns])

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}'")

    y = df[TARGET_COLUMN].map(LABEL_MAP)
    if y.isnull().any():
        raise ValueError("Loan_Status contains unexpected labels; expected only Y/N")

    x = df.drop(columns=[TARGET_COLUMN])

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(x)

    default_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier()),
        ]
    )
    default_pipeline.fit(x_train, y_train)

    k_values = [3, 5, 7]
    print("\nK experiments:")
    for k in k_values:
        experiment_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", KNeighborsClassifier(n_neighbors=k)),
            ]
        )
        experiment_pipeline.fit(x_train, y_train)
        experiment_acc = accuracy_score(y_test, experiment_pipeline.predict(x_test))
        print(f"k={k}: accuracy={experiment_acc:.4f}")

    tuned_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier()),
        ]
    )

    param_grid = {"classifier__n_neighbors": list(range(1, 26, 2))}
    grid_search = GridSearchCV(
        estimator=tuned_pipeline,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)

    print("\nBest hyperparameters from tuning:")
    print(grid_search.best_params_)
    print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

    default_results = evaluate_model("Default KNN", default_pipeline, x_test, y_test)
    tuned_results = evaluate_model("Tuned KNN", grid_search.best_estimator_, x_test, y_test)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "knn_pipeline.joblib"
    metrics_path = output_dir / "metrics.json"
    plot_path = output_dir / "knn_confusion_matrices.png"

    joblib.dump(grid_search.best_estimator_, model_path)

    metrics_payload = {
        "default": {
            "accuracy": float(default_results["accuracy"]),
            "confusion_matrix": default_results["confusion_matrix"].tolist(),
        },
        "tuned": {
            "accuracy": float(tuned_results["accuracy"]),
            "confusion_matrix": tuned_results["confusion_matrix"].tolist(),
            "best_params": grid_search.best_params_,
            "best_cv_accuracy": float(grid_search.best_score_),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ConfusionMatrixDisplay(
        confusion_matrix=default_results["confusion_matrix"],
        display_labels=["Rejected", "Approved"],
    ).plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Default KNN")

    ConfusionMatrixDisplay(
        confusion_matrix=tuned_results["confusion_matrix"],
        display_labels=["Rejected", "Approved"],
    ).plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Tuned KNN")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved tuned model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved confusion matrix plot to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate KNN loan approval model")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("../data/train.csv"),
        help="Path to Kaggle loan train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./model"),
        help="Directory to save trained model and artifacts",
    )

    args = parser.parse_args()
    main(args.data_path.resolve(), args.output_dir.resolve())
