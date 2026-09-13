from __future__ import annotations

from knn_pipeline import (
    ARTIFACT_PATH,
    PLOT_PATH,
    plot_confusion_matrices,
    save_artifact,
    train_and_evaluate,
)


def main() -> None:
    results = train_and_evaluate()

    default_metrics = results["default_metrics"]
    tuned_metrics = results["tuned_metrics"]

    print("KNN (default) accuracy:", round(default_metrics["accuracy"], 4))
    print("KNN (default) confusion matrix:", default_metrics["confusion_matrix"])

    print("K experiments (accuracy):")
    for k, score in results["k_experiments"].items():
        print(f"  k={k}: {score:.4f}")

    print("Best n_neighbors:", results["best_n_neighbors"])
    print("KNN (tuned) accuracy:", round(tuned_metrics["accuracy"], 4))
    print("KNN (tuned) confusion matrix:", tuned_metrics["confusion_matrix"])

    plot_confusion_matrices(
        default_metrics["confusion_matrix"],
        tuned_metrics["confusion_matrix"],
        output_path=PLOT_PATH,
    )

    save_artifact(
        model=results["tuned_model"],
        metadata={
            "best_n_neighbors": results["best_n_neighbors"],
            "default_metrics": default_metrics,
            "tuned_metrics": tuned_metrics,
            "k_experiments": results["k_experiments"],
            "confusion_matrix_plot": str(PLOT_PATH),
        },
        artifact_path=ARTIFACT_PATH,
    )

    print("Saved trained model to", ARTIFACT_PATH)
    print("Saved confusion matrix plot to", PLOT_PATH)


if __name__ == "__main__":
    main()
